import triton 
import triton.language as tl
import torch 
DEVICE = "cuda"
@triton.jit
def matrix_seeded_dropout(
    x_ptr,
    output_ptr,
    seeds_ptr,
    n_rows,
    n_cols,
    p,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute program ID and determine which row we're processing
    pid = tl.program_id(axis=0)
    cdive = tl.cdiv(n_cols, BLOCK_SIZE)
    row_id = pid // cdive
    col_block_id = pid % cdive
    
    # Get the seed for this row
    seed = tl.load(seeds_ptr + row_id) if row_id < n_rows else 0
    
    # Calculate starting position and offsets for this block
    block_start = col_block_id * BLOCK_SIZE
    col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Check if offsets are within bounds
    mask = (row_id < n_rows) & (col_offsets < n_cols)
    
    # Calculate memory offsets for matrix elements
    offsets = row_id * n_cols + col_offsets
    
    # Load data from input tensor
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Generate random values for dropout
    # We use a combination of seed and col_offsets to ensure different patterns per row
    # but consistent patterns when using the same seed
    random = tl.rand(seed, col_offsets)
    x_keep = random > p
    
    # Apply dropout
    output = tl.where(x_keep, x / (1 - p), 0.0)
    
    # Write back result
    tl.store(output_ptr + offsets, output, mask=mask)


def matrix_seeded_dropout_launcher(x, seeds, p):
    # Ensure inputs are contiguous
    assert x.is_contiguous()
    assert seeds.is_contiguous()
    
    # Check input dimensions
    assert x.ndim == 2, "Input tensor must be a matrix"
    assert seeds.ndim == 1, "Seeds tensor must be a vector"
    assert seeds.shape[0] == x.shape[0], "Number of seeds must match number of rows"
    
    # Get dimensions
    n_rows, n_cols = x.shape
    
    # Prepare output tensor
    output = torch.empty_like(x)
    
    # Calculate grid size - we need blocks for each row
    blocks_per_row = triton.cdiv(n_cols, 1024)
    grid = (blocks_per_row * n_rows,)
    
    # Launch kernel
    matrix_seeded_dropout[grid](
        x, output, seeds, n_rows, n_cols, p, BLOCK_SIZE=1024
    )
    
    return output


# Example usage
def test_matrix_seeded_dropout():
    # Create test input
    x = torch.randn(size=(5, 10), device=DEVICE)
    # Create different seeds for each row
    seeds = torch.tensor([123, 456, 789, 101, 202], device=DEVICE, dtype=torch.int32)
    
    # Apply dropout
    output1 = matrix_seeded_dropout_launcher(x, seeds, p=0.5)
    # Should be the same with the same seeds
    output2 = matrix_seeded_dropout_launcher(x, seeds, p=0.5)
    # Should be different with different seeds
    new_seeds = torch.tensor([999, 888, 777, 666, 555], device=DEVICE, dtype=torch.int32)
    output3 = matrix_seeded_dropout_launcher(x, new_seeds, p=0.5)
    
    # Print results
    print("Input matrix:")
    print(x)
    print("\nOutput with original seeds:")
    print(output1)
    print("\nOutput with same seeds (should match above):")
    print(output2)
    print("\nOutput with different seeds:")
    print(output3)
    
    # Verify that outputs are the same with same seeds
    assert torch.allclose(output1, output2)
    # Very unlikely that outputs would be the same with different seeds
    assert not torch.allclose(output1, output3)
    
    return output1, output2, output3

@triton.jit
def strided_matrix_seeded_dropout(
    x_ptr,
    output_ptr,
    seeds_ptr,
    n_rows,
    n_cols,
    x_row_stride,
    x_col_stride,
    out_row_stride,
    out_col_stride,
    p,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute program ID and determine which row we're processing
    pid = tl.program_id(axis=0)
    cdive = tl.cdiv(n_cols, BLOCK_SIZE)
    row_id = pid // cdive
    col_block_id = pid % cdive
    
    # Get the seed for this row
    seed = tl.load(seeds_ptr + row_id) if row_id < n_rows else 0
    
    # Calculate starting position and offsets for this block
    block_start = col_block_id * BLOCK_SIZE
    col_indices = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Check if offsets are within bounds
    mask = (row_id < n_rows) & (col_indices < n_cols)
    
    # Calculate memory offsets for matrix elements using strides
    # For a strided tensor, we need to calculate:
    # - Row offset: row_id * x_row_stride
    # - Column offset: col_indices * x_col_stride
    x_offsets = row_id * x_row_stride + col_indices * x_col_stride
    out_offsets = row_id * out_row_stride + col_indices * out_col_stride
    
    # Load data from input tensor
    x = tl.load(x_ptr + x_offsets, mask=mask)
    
    # Generate random values for dropout
    # Use a combination of seed and col_indices to ensure different patterns per row
    random = tl.rand(seed, col_indices)
    x_keep = random > p
    
    # Apply dropout
    output = tl.where(x_keep, x / (1 - p), 0.0)
    
    # Write back result using output strides
    tl.store(output_ptr + out_offsets, output, mask=mask)


def strided_matrix_seeded_dropout_launcher(x, seeds, p):
    # Check input dimensions
    assert x.ndim == 2, "Input tensor must be a matrix"
    assert seeds.ndim == 1, "Seeds tensor must be a vector"
    assert seeds.shape[0] == x.shape[0], "Number of seeds must match number of rows"
    
    # Get dimensions
    n_rows, n_cols = x.shape
    
    # Get strides in elements (not bytes)
    # For PyTorch tensors, strides are in elements not bytes
    x_row_stride = x.stride(0)
    x_col_stride = x.stride(1)
    
    # Prepare output tensor
    output = torch.empty_like(x)
    out_row_stride = output.stride(0)
    out_col_stride = output.stride(1)
    
    # Calculate grid size - we need blocks for each row
    blocks_per_row = triton.cdiv(n_cols, 1024)
    grid = (blocks_per_row * n_rows,)
    
    # Launch kernel
    strided_matrix_seeded_dropout[grid](
        x, 
        output, 
        seeds, 
        n_rows, 
        n_cols, 
        x_row_stride,
        x_col_stride,
        out_row_stride,
        out_col_stride,
        p, 
        BLOCK_SIZE=1024
    )
    
    return output


# Example usage to demonstrate stride handling
def test_strided_dropout():
    # Create a 10x8 matrix
    x = torch.randn(10, 8, device=DEVICE)
    
    # Create a non-contiguous view by transposing
    x_transposed = x.t()  # This creates a non-contiguous tensor with different strides
    
    # Create seeds tensor
    seeds = torch.arange(8, device=DEVICE, dtype=torch.int32)
    
    # Apply dropout to transposed (non-contiguous) tensor
    output = strided_matrix_seeded_dropout_launcher(x_transposed, seeds, p=0.5)
    
    # Verify shapes
    print(f"Input shape: {x_transposed.shape}")
    print(f"Input strides: {x_transposed.stride()}")
    print(f"Output shape: {output.shape}")
    
    # Apply again to show consistency with same seeds
    output2 = strided_matrix_seeded_dropout_launcher(x_transposed, seeds, p=0.5)
    
    # These should be equal if using the same seeds
    print(f"Outputs match: {torch.allclose(output, output2)}")
    
    return output

test_matrix_seeded_dropout()
test_strided_dropout()
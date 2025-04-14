import triton
import triton.language as tl
import torch
import math

@triton.jit
def sparse_jl_transform_kernel(
    x_ptr,                      # Input matrix pointer [n_samples, n_features]
    output_ptr,                 # Output matrix pointer [n_samples, n_components]
    n_samples,                  # Number of samples
    n_features,                 # Original dimensionality
    n_components,               # Target dimensionality
    sparsity,                   # Sparsity factor (how many features to consider per component)
    seed,                       # Random seed
    x_row_stride,               # Row stride of input
    x_col_stride,               # Column stride of input
    out_row_stride,             # Row stride of output
    out_col_stride,             # Column stride of output
    BLOCK_SIZE: tl.constexpr,   # Block size for parallelization
):
    # Get program ID and calculate which rows we're processing
    pid = tl.program_id(axis=0)
    sample_id = pid
    
    # Make sure we're within bounds
    if sample_id >= n_samples:
        return
    
    # Process one sample (row) at a time
    for component_id in range(0, n_components):
        # Initialize accumulator for this component
        acc = 0.0
        
        # Derive a unique seed for this component
        component_seed = seed + component_id * 1237
        
        # Generate sparse pattern - determine which features to use for this component
        # We'll use a deterministic approach based on the seed and component_id
        num_nonzero = tl.math.min(sparsity, n_features)
        
        # Process each non-zero element
        for idx in range(0, num_nonzero):
            # Generate pseudo-random feature index
            # This is a simple hash function to deterministically select features
            feature_seed = component_seed + idx * 2749
            feature_idx = tl.rand(feature_seed, 0) * n_features
            feature_idx = tl.math.min(feature_idx, n_features - 1)
            
            # Convert to integer
            feature_idx = tl.math.floor(feature_idx).to(tl.int32)
            
            # Generate pseudo-random value (-1, +1) using another seed
            value_seed = feature_seed + 104729
            rand_val = tl.rand(value_seed, 0)
            # Generate sparse value {-1/sqrt(s), +1/sqrt(s)} where s is number of non-zeros
            sparse_val = tl.where(rand_val > 0.5, 1.0, -1.0) / tl.math.sqrt(float(num_nonzero))
            
            # Load the feature value for this sample
            x_offset = sample_id * x_row_stride + feature_idx * x_col_stride
            x_val = tl.load(x_ptr + x_offset)
            
            # Accumulate the contribution
            acc += x_val * sparse_val
        
        # Write the result for this component
        out_offset = sample_id * out_row_stride + component_id * out_col_stride
        tl.store(output_ptr + out_offset, acc)


def sparse_jl_transform(x, n_components, sparsity=None, seed=42):
    """
    Apply a sparse Johnson-Lindenstrauss transform to the input data.
    
    Parameters:
    ----------
    x : torch.Tensor
        Input tensor of shape [n_samples, n_features]
    n_components : int
        Target dimensionality
    sparsity : int, optional
        Number of non-zero elements per component. If None, will use log(n_features)
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    -------
    torch.Tensor
        Transformed data of shape [n_samples, n_components]
    """
    assert x.ndim == 2, "Input tensor must be 2D (samples × features)"
    
    # Get dimensions
    n_samples, n_features = x.shape
    
    # Set default sparsity if not provided
    if sparsity is None:
        sparsity = max(1, int(math.log(n_features)))
    
    # Ensure sparsity is valid
    sparsity = min(sparsity, n_features)
    
    # Create output tensor
    output = torch.empty((n_samples, n_components), dtype=x.dtype, device=x.device)
    
    # Get stride information
    x_row_stride = x.stride(0)
    x_col_stride = x.stride(1)
    out_row_stride = output.stride(0)
    out_col_stride = output.stride(1)
    
    # Launch kernel
    grid = (n_samples,)
    sparse_jl_transform_kernel[grid](
        x, 
        output, 
        n_samples, 
        n_features, 
        n_components, 
        sparsity, 
        seed,
        x_row_stride,
        x_col_stride,
        out_row_stride,
        out_col_stride,
        BLOCK_SIZE=1024
    )
    
    return output


def test_sparse_jl_transform():
    # Create test data
    n_samples = 1000
    n_features = 10000
    n_components = 100
    sparsity = 20  # Each component considers 20 features
    
    # Create random input data
    x = torch.randn(n_samples, n_features, device="cuda")
    
    # Apply the transform
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    output = sparse_jl_transform(x, n_components, sparsity, seed=42)
    end_time.record()
    
    torch.cuda.synchronize()
    elapsed_time = start_time.elapsed_time(end_time)
    
    # Check results
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Time taken: {elapsed_time:.2f} ms")
    
    # Verify determinism - should be identical with same seed
    output2 = sparse_jl_transform(x, n_components, sparsity, seed=42)
    print(f"Outputs match with same seed: {torch.allclose(output, output2)}")
    
    # Verify different seed gives different result
    output3 = sparse_jl_transform(x, n_components, sparsity, seed=123)
    print(f"Outputs differ with different seed: {not torch.allclose(output, output3)}")
    
    # Calculate the average norm ratio to check if the transform approximately preserves distances
    x_norms = torch.norm(x, dim=1)
    output_norms = torch.norm(output, dim=1)
    avg_norm_ratio = (output_norms / x_norms).mean().item()
    print(f"Average norm ratio: {avg_norm_ratio:.4f}")
    
    return output, elapsed_time
test_sparse_jl_transform()
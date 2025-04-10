import torch

import triton
import triton.language as tl

DEVICE = torch.device('cuda')

@triton.jit 
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr, num_stages: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)

    for idx in range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + idx * input_row_stride

        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets

        mask = col_offsets < n_cols 

        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

        row_min_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_min_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator

        output_row_start_ptr = output_ptr + idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)

def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"

def is_cdna():
    return triton.runtime.driver.active.get_current_target().arch == "gfx90a"

def softmax(x):
    n_rows, n_cols = x.shape

    # The block size of each loop iteration is the smallest power of two greater than the number of columns in `x`
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # Use 8 warps per block (256 threads)
    num_warps = 8

    # Use 2 stages for software pipelining
    num_stages = 2

    # Allocate output
    y = torch.empty_like(x)

    # pre-compile kernel to get register usage and compute thread occupancy.
    kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE,
                                   num_stages=num_stages, num_warps=num_warps, grid=(1,))
    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared

    # Hardcoded values for CUDA devices
    NUM_SM = 108  # Typical for modern GPUs
    NUM_REGS = 65536  # Typical register file size
    WARP_SIZE = 32  # Standard CUDA warp size
    MAX_NUM_THREADS = 2048  # Typical max threads per SM

    # Calculate occupancy
    occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, MAX_NUM_THREADS // (WARP_SIZE * num_warps))
    num_programs = NUM_SM * occupancy
    num_programs = min(num_programs, n_rows)

    # Create a number of persistent programs.
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_stages=num_stages)
    return y

# Test the implementation
torch.manual_seed(0)
x = torch.randn(1823, 781, device=DEVICE)
y_triton = softmax(x)
y_torch = torch.softmax(x, axis=1)
assert torch.allclose(y_triton, y_torch, rtol=1e-3, atol=1e-3), (y_triton, y_torch)
print("Test passed!")

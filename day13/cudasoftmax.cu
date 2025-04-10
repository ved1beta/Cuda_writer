#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>


__global__ void softmax_kernel(float *output, const float *input, int num_rows, int num_cols){

    extern __shared__ float shared[];
    int tid = threadIdx.x; 
    int row = blockDim.x;
    int row_strid = row * num_cols ;

    float max_val = -INFINITY ;
    for (int  i = tid; i < num_cols; i+= blockDim.x)
    { 
        float val = input[row_strid + i];
        if (val > max_val) max_val = val;
    }

    shared[tid] = max_val;
    __syncthreads();

    // assume blockdim is power of 2 : ) 
     for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared[tid + s] > shared[tid]) {
                shared[tid] = shared[tid + s];
            }
        }
        __syncthreads();
    }
    float row_max = shared[0];
    __syncthreads();


    float exp_sum = 0.0f;
    for (int i = tid; i < num_cols; i += blockDim.x) {
        float val = input[row_strid + i];
        exp_sum += expf(val - row_max);
    }

    // Reduce sum across threads
    shared[tid] = exp_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    float row_sum = shared[0];
    __syncthreads();

    // Step 3: Compute softmax and write to output
    for (int i = tid; i < num_cols; i += blockDim.x) {
        float val = input[row_strid + i];
        output[row_strid + i] = expf(val - row_max) / row_sum;
    }
}

// Helper function to check CUDA errors
#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)
void check_cuda_error(cudaError_t result, const char *func, const char *file, int line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",
                file, line, result, cudaGetErrorString(result), func);
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Define matrix dimensions
    const int num_rows = 1000;
    const int num_cols = 1000;
    const size_t size = num_rows * num_cols * sizeof(float);

    // Allocate host memory
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size);

    // Initialize input with random values
    srand(time(NULL));
    for (int i = 0; i < num_rows * num_cols; i++) {
        h_input[i] = (float)rand() / RAND_MAX;
    }

    // Allocate device memory
    float *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, size));

    // Copy input data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int block_size = 256; // Use a power of 2
    int grid_size = num_rows;
    size_t shared_mem_size = block_size * sizeof(float);

    softmax_kernel<<<grid_size, block_size, shared_mem_size>>>(d_output, d_input, num_rows, num_cols);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    // Verify results (simple check)
    float sum = 0.0f;
    for (int i = 0; i < num_cols; i++) {
        sum += h_output[i];
    }
    printf("Sum of first row: %f (should be close to 1.0)\n", sum);

    // Free memory
    free(h_input);
    free(h_output);
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));

    return 0;
}

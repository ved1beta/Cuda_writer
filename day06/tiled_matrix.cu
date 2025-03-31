#include <stdio.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

// Tile size for the CUDA kernel
#define TILE_SIZE 32

// Kernel for tiled matrix multiplication
__global__ void tiledMatrixMultiplication(float* A, float* B, float* C, int n) {
    // Shared memory for the tiles
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];
    
    // Calculate row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Thread indices within the tile
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Variable to accumulate result
    float result = 0.0f;
    
    // Loop over all tiles required to compute the output element
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load elements from matrix A into shared memory
        if (row < n && t * TILE_SIZE + tx < n) {
            A_tile[ty][tx] = A[row * n + t * TILE_SIZE + tx];
        } else {
            A_tile[ty][tx] = 0.0f;
        }
        
        // Load elements from matrix B into shared memory
        if (col < n && t * TILE_SIZE + ty < n) {
            B_tile[ty][tx] = B[(t * TILE_SIZE + ty) * n + col];
        } else {
            B_tile[ty][tx] = 0.0f;
        }
        
        // Synchronize to ensure all data is loaded before computation
        __syncthreads();
        
        // Multiply the two tiles and accumulate the results
        for (int k = 0; k < TILE_SIZE; k++) {
            result += A_tile[ty][k] * B_tile[k][tx];
        }
        
        // Synchronize to ensure all computations are done before loading new tiles
        __syncthreads();
    }
    
    // Write the result to global memory
    if (row < n && col < n) {
        C[row * n + col] = result;
    }
}

// Kernel for generating random matrices
__global__ void generateRandomMatrix(float* matrix, int n, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < n && idy < n) {
        int index = idy * n + idx;
        curandState state;
        curand_init(seed, index, 0, &state);
        matrix[index] = curand_uniform(&state);
    }
}

int main() {
    int n = 1024; // Matrix size (n x n)
    size_t matrixSize = n * n * sizeof(float);
    
    // Host matrices
    float *h_C = (float*)malloc(matrixSize);
    
    // Device matrices
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, matrixSize);
    cudaMalloc((void**)&d_B, matrixSize);
    cudaMalloc((void**)&d_C, matrixSize);
    
    // Define block and grid dimensions for matrix generation
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);
    
    // Generate random matrices A and B on the device
    unsigned long seed = time(NULL);
    generateRandomMatrix<<<gridSize, blockSize>>>(d_A, n, seed);
    generateRandomMatrix<<<gridSize, blockSize>>>(d_B, n, seed + 1);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start event
    cudaEventRecord(start);
    
    // Execute the tiled matrix multiplication kernel
    tiledMatrixMultiplication<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);
    
    // Record stop event and synchronize
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate and print elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);
    
    // Copy result matrix back to host
    cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost);
    
    // Verify the result (optional - just printing a small portion)
    printf("Sample of the result matrix (top-left 5x5):\n");
    for (int i = 0; i < 5 && i < n; i++) {
        for (int j = 0; j < 5 && j < n; j++) {
            printf("%f ", h_C[i * n + j]);
        }
        printf("\n");
    }
    
    // Free resources
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}
#include <iostream>
#include <cuda_runtime.h>

__global__ void matadd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // More explicit bounds checking
    if (i < N && j < N) {
        C[i*N+j] = A[i*N+j] + B[i*N+j];
    }
}

int main() {
    const int N = 10;
    size_t matrix_size = N * N * sizeof(float);

    // Host matrices
    float *A, *B, *C;
    A = (float *)malloc(matrix_size);
    B = (float *)malloc(matrix_size);
    C = (float *)malloc(matrix_size);

    // Initialize matrices
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = 1.0f;
            B[i * N + j] = 2.0f;
            C[i * N + j] = 0.0f;
        }
    }

    // Device pointers
    float *d_a, *d_b, *d_c;

    // CUDA Memory Allocation
    cudaMalloc((void **)&d_a, matrix_size);
    cudaMalloc((void **)&d_b, matrix_size);
    cudaMalloc((void **)&d_c, matrix_size);

    // Memory Transfer to Device
    cudaMemcpy(d_a, A, matrix_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, matrix_size, cudaMemcpyHostToDevice);

    // Kernel Configuration
    dim3 dimBlock(32, 16);
    dim3 dimGrid(
        (N + dimBlock.x - 1) / dimBlock.x, 
        (N + dimBlock.y - 1) / dimBlock.y
    );

    // Kernel Launch with Error Checking
    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) {
        std::cerr << "Error before kernel launch: " 
                  << cudaGetErrorString(kernelError) << std::endl;
        return -1;
    }

    // Launch Kernel
    matadd<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);

    // Check Kernel Execution
    kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) {
        std::cerr << "Kernel launch error: " 
                  << cudaGetErrorString(kernelError) << std::endl;
        return -1;
    }

    // Copy Result Back
    cudaMemcpy(C, d_c, matrix_size, cudaMemcpyDeviceToHost);

    // Synchronize and Check
    cudaError_t syncError = cudaDeviceSynchronize();
    if (syncError != cudaSuccess) {
        std::cerr << "Synchronization error: " 
                  << cudaGetErrorString(syncError) << std::endl;
        return -1;
    }

    // Verify Results (Optional)
    bool valid = true;
    for (int i = 0; i < N * N; i++) {
        if (C[i] != 3.0f) {
            valid = false;
            break;
        }
    }
    std::cout << "Computation " 
              << (valid ? "Successful" : "Failed") << std::endl;

    // Cleanup
    free(A);
    free(B);
    free(C);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
#include <stdio.h>

// CUDA kernel for vector addition
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    // Calculate global thread ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Ensure we don't access beyond array bounds
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    // Vector size
    int n = 1000000;
    size_t size = n * sizeof(float);
    
    // Host vectors
    float *h_a, *h_b, *h_c;
    
    // Device vectors
    float *d_a, *d_b, *d_c;
    
    // Allocate memory for host vectors
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);
    
    // Initialize host vectors
    for (int i = 0; i < n; i++) {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate memory for device vectors
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Define thread block and grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    
    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // Verify result
    for (int i = 0; i < n; i++) {
        if (fabs(h_a[i] + h_b[i] - h_c[i]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            break;
        }
    }
    
    printf("Test PASSED\n");
    
    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}
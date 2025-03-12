#include <iostream>

// Function to convert milliseconds to a readable format
void printTime(float ms) {
    if (ms < 1.0) {
        printf("Time: %.3f microseconds\n", ms * 1000.0f);
    } else if (ms < 1000.0) {
        printf("Time: %.3f milliseconds\n", ms);
    } else {
        printf("Time: %.3f seconds\n", ms / 1000.0f);
    }
}

// Function to print memory usage in a readable format
void printMemoryUsage(size_t bytes) {
    const double kb = 1024.0;
    const double mb = kb * 1024.0;
    const double gb = mb * 1024.0;

    if (bytes < kb) {
        printf("Memory usage: %zu bytes\n", bytes);
    } else if (bytes < mb) {
        printf("Memory usage: %.2f KB\n", bytes / kb);
    } else if (bytes < gb) {
        printf("Memory usage: %.2f MB\n", bytes / mb);
    } else {
        printf("Memory usage: %.2f GB\n", bytes / gb);
    }
}

__global__ void vectorMatrixMult(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float sharedB[];
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        sharedB[j] = B[j];
    }
    __syncthreads();
    if (i < N) {
      float sum=0.0f;
      for (int j = 0; j < N; j++) {
         sum += A[i*N+j]* sharedB[j];
      }
      C[i]=sum;
    }
}

int main() {
    // Initialize the matrix
    const int N = 100;
    float *A, *B, *C;
    size_t totalMemoryUsage = 0;

    // Initialize the input matrices
    A = (float *)malloc(N*N*sizeof(float));
    B = (float *)malloc(N*sizeof(float));
    C = (float *)malloc(N*sizeof(float));
    
    // Track host memory usage
    totalMemoryUsage += N*N*sizeof(float);
    totalMemoryUsage += N*sizeof(float);
    totalMemoryUsage += N*sizeof(float);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = 1.0f;
        }
        B[i] = 2.0f;
        C[i] = 0.0f;
    }

    float *d_a, *d_b, *d_c;
    size_t deviceMemoryUsage = 0;
    
    // Allocate device memory
    cudaMalloc(&d_a, N*N*sizeof(float));
    cudaMalloc(&d_b, N*sizeof(float));
    cudaMalloc(&d_c, N*sizeof(float));
    
    // Track device memory usage
    deviceMemoryUsage += N*N*sizeof(float);
    deviceMemoryUsage += N*sizeof(float);
    deviceMemoryUsage += N*sizeof(float);
    
    // Copy data to device
    cudaMemcpy(d_a, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B, N*sizeof(float), cudaMemcpyHostToDevice);
    
    // Setup CUDA timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Configure kernel parameters
    int blocksize = 256;
    int gridsize = (N + blocksize - 1) / blocksize;
    
    // Start timing
    cudaEventRecord(start);
    
    // Launch kernel
    vectorMatrixMult<<<gridsize, blocksize>>>(d_a, d_b, d_c, N);
    
    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back to host
    cudaMemcpy(C, d_c, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Print memory usage statistics
    printf("Host memory usage: ");
    printMemoryUsage(totalMemoryUsage);
    printf("Device memory usage: ");
    printMemoryUsage(deviceMemoryUsage);
    printf("Total memory usage: ");
    printMemoryUsage(totalMemoryUsage + deviceMemoryUsage);
    
    // Print timing information
    printf("Kernel execution time: ");
    printTime(milliseconds);
    
    // Print results (you can keep your existing print code)
    printf("A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", A[i * N + j]);
        }
        printf("\n");
    }

    printf("C:\n");
    for (int i = 0; i < N; i++) {
        printf("%.2f ", C[i]);
    }
    printf("\n");
    
    printf("B:\n");
    for (int i = 0; i < N; i++) {
        printf("%.2f ", B[i]);
    }
    printf("\n");

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(A);
    free(B);
    free(C);
    
    return 0;
}
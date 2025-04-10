#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Error checking macro
#define cudaCheckError() { \
    cudaError_t e=cudaGetLastError(); \
    if(e!=cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

// CUDA kernel for vector addition
__global__ void add(float* A, float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i < n) {
        C[i] = B[i] + A[i];
    }
}

// Function to print an array as a horizontal bar chart
void printBarChart(const char* label, float* data, int n) {
    printf("\n%s:\n", label);
    
    // Find the maximum value for scaling
    float max_val = 0;
    for (int i = 0; i < n; i++) {
        if (fabs(data[i]) > max_val) {
            max_val = fabs(data[i]);
        }
    }
    
    // Scale factor - how many characters per unit
    const int MAX_WIDTH = 50;
    float scale = max_val > 0 ? MAX_WIDTH / max_val : 1;
    
    // Print each value as a bar
    for (int i = 0; i < n; i++) {
        // Print index and value
        printf("[%2d] %6.2f |", i, data[i]);
        
        // Print the bar
        int bar_length = (int)(fabs(data[i]) * scale);
        for (int j = 0; j < bar_length; j++) {
            printf("#");
        }
        printf("\n");
    }
    printf("\n");
}

// Function to visualize the addition operation
void visualizeAddition(float* A, float* B, float* C, int n) {
    printf("\nVisualization of A + B = C:\n");
    printf("---------------------------\n");
    
    for (int i = 0; i < n; i++) {
        printf("[%2d] %6.2f + %6.2f = %6.2f\n", i, A[i], B[i], C[i]);
    }
    printf("\n");
}

int main() {
    const int N = 10;
    float A[N], B[N], C[N];

    // Initialize input arrays with some interesting patterns
    for (int i = 0; i < N; i++) {
        A[i] = i * 1.0f;          // Linear increase
        B[i] = N - i * 0.5f;      // Linear decrease
    }

    float *d_a, *d_b, *d_c;

    // Allocate device memory
    cudaMalloc(&d_a, N * sizeof(float));
    cudaCheckError();
    cudaMalloc(&d_b, N * sizeof(float));
    cudaCheckError();
    cudaMalloc(&d_c, N * sizeof(float));
    cudaCheckError();

    // Copy data to device
    cudaMemcpy(d_a, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(d_b, B, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError();

    // Print execution information
    printf("CUDA Vector Addition Visualization\n");
    printf("==================================\n");
    printf("Vector size: %d elements\n", N);
    
    // Launch kernel with timing
    int blocksize = 256;
    int gridsize = (N + blocksize - 1) / blocksize;
    
    printf("Kernel launch configuration: grid=%d, block=%d\n\n", gridsize, blocksize);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start time
    cudaEventRecord(start);
    
    // Launch kernel
    add<<<gridsize, blocksize>>>(d_a, d_b, d_c, N);
    cudaCheckError();
    
    // Record stop time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %.3f milliseconds\n", milliseconds);
    
    // Copy result back to host
    cudaMemcpy(C, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();

    // Visualize input arrays and result
    printBarChart("Array A (Input 1)", A, N);
    printBarChart("Array B (Input 2)", B, N);
    printBarChart("Array C (Result of A + B)", C, N);
    
    // Visualize the addition operation
    visualizeAddition(A, B, C, N);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
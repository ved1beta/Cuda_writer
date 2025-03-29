
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Error checking macro
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(error)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Improved softmax kernel using shared memory and parallel reduction
__global__ void softmaxKernel(float* input, float* output, int size) {
    extern __shared__ float sharedData[];
    
    // Shared memory for storing maximum and sum
    float* sharedMax = sharedData;
    float* sharedSum = &sharedData[blockDim.x];
    
    const int tid = threadIdx.x;
    const int blockOffset = blockIdx.x * size;
    
    // Initialize local maximum to the smallest possible float value
    float localMax = -INFINITY;
    
    // Each thread finds its local maximum
    for (int i = tid; i < size; i += blockDim.x) {
        localMax = fmaxf(localMax, input[blockOffset + i]);
    }
    
    // Store local maximum to shared memory
    sharedMax[tid] = localMax;
    __syncthreads();
    
    // Perform parallel reduction to find the block's maximum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedMax[tid] = fmaxf(sharedMax[tid], sharedMax[tid + stride]);
        }
        __syncthreads();
    }
    
    // At this point, sharedMax[0] contains the maximum value for this block
    float blockMax = sharedMax[0];
    
    // Compute local sum of exponentials
    float localSum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
        localSum += expf(input[blockOffset + i] - blockMax);
    }
    
    // Store local sum to shared memory
    sharedSum[tid] = localSum;
    __syncthreads();
    
    // Perform parallel reduction to sum all exponentials
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedSum[tid] += sharedSum[tid + stride];
        }
        __syncthreads();
    }
    
    // At this point, sharedSum[0] contains the sum of exponentials for this block
    float blockSum = sharedSum[0];
    
    // Each thread computes softmax outputs for its assigned elements
    for (int i = tid; i < size; i += blockDim.x) {
        output[blockOffset + i] = expf(input[blockOffset + i] - blockMax) / blockSum;
    }
}

// CPU version of softmax for verification
void softmaxCPU(float* input, float* output, int size) {
    // Find maximum value
    float maxVal = input[0];
    for (int i = 1; i < size; i++) {
        maxVal = fmaxf(maxVal, input[i]);
    }
    
    // Compute sum of exponentials
    float sumExp = 0.0f;
    for (int i = 0; i < size; i++) {
        sumExp += expf(input[i] - maxVal);
    }
    
    // Compute softmax values
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - maxVal) / sumExp;
    }
}

// Function to verify results
bool verifySoftmax(float* gpuResult, float* cpuResult, int size, float tolerance) {
    bool correct = true;
    float maxDiff = 0.0f;
    
    for (int i = 0; i < size; i++) {
        float diff = fabsf(gpuResult[i] - cpuResult[i]);
        maxDiff = fmaxf(maxDiff, diff);
        
        if (diff > tolerance) {
            printf("Error at index %d: GPU = %f, CPU = %f, diff = %f\n", 
                   i, gpuResult[i], cpuResult[i], diff);
            correct = false;
        }
    }
    
    printf("Maximum difference: %f\n", maxDiff);
    return correct;
}

// Test function that runs both CPU and GPU implementations
void testSoftmax(int size, int numBatches) {
    printf("Testing softmax with size = %d, batches = %d\n", size, numBatches);
    
    size_t totalSize = size * numBatches;
    size_t memSize = sizeof(float) * totalSize;
    
    // Allocate host memory
    float* h_input = (float*)malloc(memSize);
    float* h_output_gpu = (float*)malloc(memSize);
    float* h_output_cpu = (float*)malloc(memSize);
    
    // Initialize input with random values
    srand(42);
    for (int i = 0; i < totalSize; i++) {
        h_input[i] = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;  // Range: -5 to 5
    }
    
    // Allocate device memory
    float* d_input;
    float* d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, memSize));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, memSize));
    
    // Copy input to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, memSize, cudaMemcpyHostToDevice));
    
    // Configure kernel
    int blockSize = 256;
    int sharedMemSize = 2 * blockSize * sizeof(float);  // For max and sum
    
    // Launch kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    softmaxKernel<<<numBatches, blockSize, sharedMemSize>>>(d_input, d_output, size);
    cudaEventRecord(stop);
    
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Calculate kernel execution time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU execution time: %.3f ms\n", milliseconds);
    
    // Copy results back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output_gpu, d_output, memSize, cudaMemcpyDeviceToHost));
    
    // Compute CPU softmax for verification
    clock_t cpu_start = clock();
    for (int batch = 0; batch < numBatches; batch++) {
        softmaxCPU(h_input + batch * size, h_output_cpu + batch * size, size);
    }
    clock_t cpu_end = clock();
    double cpu_time = 1000.0 * (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;
    printf("CPU execution time: %.3f ms\n", cpu_time);
    
    // Verify results
    bool correct = true;
    for (int batch = 0; batch < numBatches; batch++) {
        if (!verifySoftmax(h_output_gpu + batch * size, h_output_cpu + batch * size, size, 1e-5f)) {
            printf("Verification failed for batch %d\n", batch);
            correct = false;
            break;
        }
    }
    
    if (correct) {
        printf("All results match within tolerance!\n");
        printf("Speedup: %.2fx\n", cpu_time / milliseconds);
    }
    
    // Free memory
    free(h_input);
    free(h_output_gpu);
    free(h_output_cpu);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    // Test with different sizes and batch counts
    printf("======= CUDA Softmax Test =======\n");
    
    // Small size test
    testSoftmax(128, 1);
    printf("\n");
    
    // Medium size test
    testSoftmax(1024, 10);
    printf("\n");
    
    // Large size test
    testSoftmax(4096, 32);
    
    return 0;
}
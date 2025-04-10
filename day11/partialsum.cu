#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>

// PART 1: PARALLEL REDUCTION IMPLEMENTATIONS

// 1. Naive parallel reduction
__global__ void reduceNaive(float* input, float* output, int size) {
    // TODO: Implement naive reduction with thread divergence
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i= threadIdx.y + blockDim.y * blockIdx.y;

    sdata[tid] = (i < size ) ? input[i]: 0 ;
    __syncthreads();

    for(int s = 1 ; s < blockDim.x; s*=2){
        if(tid % 2 == 0 ){
            if (tid + s < blockDim.x){
                sdata[tid] += sdata[tid+s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// 2. Optimized reduction - Avoids divergent branching
__global__ void reduceSharedMemoryEfficient(float* input, float* output, int size) {
    extern __shared__ float sdata[];
    
    // Each thread loads one element from global to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (2 * blockDim.x) + tid;
 
    sdata[tid] = 0;
    if (i < size) sdata[tid] = input[i];
    if (i + blockDim.x < size) sdata[tid] += input[i + blockDim.x];
    __syncthreads();
    
    // Do reduction in shared memory with sequential addressing
    for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Function to launch a reduction kernel
float launchReduction(float* d_input, int size, int blockSize, void (*reductionKernel)(float*, float*, int)) {
    // TODO: Set up grid dimensions, allocate memory for results, 
    // launch kernel, and handle multi-block reduction
}

// PART 2: PARALLEL SCAN IMPLEMENTATIONS

// 1. Inclusive scan (prefix sum)
__global__ void inclusiveScanKernel(float* input, float* output, int size) {
    // TODO: Implement inclusive scan
}

// 2. Exclusive scan
__global__ void exclusiveScanKernel(float* input, float* output, int size) {
    // TODO: Implement exclusive scan
}

// Function to launch scan kernels
void launchScan(float* d_input, float* d_output, int size, int blockSize, bool isInclusive) {
    // TODO: Configure and launch appropriate scan kernel
}

// PART 3: PERFORMANCE ANALYSIS FUNCTIONS

// Function to measure kernel execution time
float measureKernelTime(void (*kernelLauncher)(float*, float*, int, int), 
                       float* d_input, float* d_output, int size, int blockSize) {
    // TODO: Implement timing code using CUDA events
}

// Function to calculate work efficiency
float calculateWorkEfficiency(int sequentialOperations, int parallelOperations) {
    // TODO: Calculate and return the work efficiency ratio
}

// Function to analyze thread divergence
void analyzeThreadDivergence(int size, int blockSize) {
    // TODO: Implement analysis of thread divergence impact on performance
}

// Function to experiment with different configurations
void experimentWithConfigurations(float* h_input, int size) {
    // TODO: Test performance with different block sizes and input sizes
}
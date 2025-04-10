#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>

__global__ void add(float* A , float* B , float* C , int n  ){
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    if(i < n){
        C[i] = B[i] + A[i];
    }
}

int main(){
    const int N = 10;
    float A[N], B[N], C[N];

    // Initialize input arrays
    for (int i = 0; i < N; i++) {
        A[i] = i * 1.0f;  // Some sample values
        B[i] = i * 2.0f;
    }

    float *d_a , *d_b , *d_c ;

    cudaMalloc(&d_a , N * sizeof(float));
    cudaMalloc(&d_b , N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    cudaMemcpy(d_a , A , N* sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b , B , N* sizeof(float), cudaMemcpyHostToDevice);

    int blocksize = 256 ; 
    int gridsize = (N+ blocksize -1)/blocksize; 

    add<<<gridsize , blocksize>>>(d_a , d_b , d_c , N);

    cudaError_t kernelError = cudaGetLastError();
    if (kernelError != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(kernelError) << std::endl;
        return -1;
    }

    // Copy result back to host
    cudaMemcpy(C, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Check for any CUDA errors
    cudaError_t syncError = cudaGetLastError();
    if (syncError != cudaSuccess) {
        std::cerr << "CUDA error after kernel execution: " << cudaGetErrorString(syncError) << std::endl;
        return -1;
    }

    // Print results
    std::cout << "Vector Addition Results:" << std::endl;
    for (int i = 0; i < N; i++) {
        std::cout << A[i] << " + " << B[i] << " = " << C[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
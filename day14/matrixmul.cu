#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N)(((M)+(N)-1)/(N))

template <const int BM  , const int BN , const int BK, const int TM , const int TN> 
__global__ void  __launch_bounds__((BM * BN )/(TM * TN), 1)
        segemm2d(int M , int N , int K , float alpha , const float *A, const float *B , float beta, float *C ){


            const uint cRow = blockIdx.y;
            const uint cCol = blockIdx.x;

            const uint totalResultsBlocktile = BM * BN;

            const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

            assert(numThreadsBlocktile == blockDim.x);

            const int threadCol = threadIdx.x % (BN / TN);
            const int threadRow = threadIdx.x / (BN / TN);

            __shared__ float As[BM * BK];
            __shared__ float Bs[BK * BN];

 //            Move blocktile to beginning of A's row and B's column
            A += cRow * BM * K;
            B += cCol * BN;
            C += cRow * BM * N + cCol * BN;

            const uint innerRowA = threadIdx.x / BK; 
            const uint innerColA = threadIdx.x % BK;

            const uint strideA = numThreadsBlocktile / BK;
            const uint innerRowB = threadIdx.x / BN;
            const uint innerColB = threadIdx.x % BN;
            // for both As and Bs we want each load to span the full column-width, for
            // better GMEM coalescing (as opposed to spanning full row-width and iterating
            // across columns)
            const uint strideB = numThreadsBlocktile / BN;

            float threadResults[TM * TN] = {0.0};
  // register caches for As and Bs
            float regM[TM] = {0.0};
            float regN[TN] = {0.0};

            for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
            for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
              As[(innerRowA + loadOffset) * BK + innerColA] =
                  A[(innerRowA + loadOffset) * K + innerColA];
            }
            for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
              Bs[(innerRowB + loadOffset) * BN + innerColB] =
                  B[(innerRowB + loadOffset) * N + innerColB];
            }
            __syncthreads();

            A += BK;
            B += BK * N;

            for(uint dotIdx = 0 ; dotIdx < BK ; ++dotIdx){
                for(uint i = 0 ; i < TM ; ++i){
                    regM[i] = As[(threadRow * TM + i)*BK + dotIdx];

                }
                for (uint i = 0; i < TN; ++i) {
                    regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
                  }
                  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                      threadResults[resIdxM * TN + resIdxN] +=
                          regM[resIdxM] * regN[resIdxN];
                    }
                  }
            }
                __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          alpha * threadResults[resIdxM * TN + resIdxN] +
          beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
    }
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
    // Matrix dimensions
    const int M = 1024;  // Rows of A and C
    const int N = 1024;  // Columns of B and C
    const int K = 1024;  // Columns of A and rows of B

    // Block and thread tile sizes
    const int BM = 128;  // Block size for M dimension
    const int BN = 128;  // Block size for N dimension
    const int BK = 32;   // Block size for K dimension
    const int TM = 8;    // Thread tile size for M dimension
    const int TN = 8;    // Thread tile size for N dimension

    // Allocate host memory
    float *h_A = (float *)malloc(M * K * sizeof(float));
    float *h_B = (float *)malloc(K * N * sizeof(float));
    float *h_C = (float *)malloc(M * N * sizeof(float));
    float *h_C_ref = (float *)malloc(M * N * sizeof(float));

    // Initialize matrices with random values
    srand(time(NULL));
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < M * N; i++) h_C[i] = 0.0f;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, M * K * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, M * N * sizeof(float)));

    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    float alpha = 1.0f;
    float beta = 0.0f;

    // Warmup
    segemm2d<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Time the kernel
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    // Run kernel
    segemm2d<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute reference using cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_ref, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify results
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float diff = fabs(h_C[i] - h_C_ref[i]);
        if (diff > max_diff) max_diff = diff;
    }
    printf("Maximum difference between custom and cuBLAS: %f\n", max_diff);

    // Cleanup
    cublasDestroy(handle);
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);

    return 0;
}


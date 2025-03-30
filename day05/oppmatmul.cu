#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cassert>  // Added this include for assert()
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// Error checking macro
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// The kernel from your code
template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;
  const uint totalResultsBlocktile = BM * BN;
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);
  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x);
  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);
  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];
  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;
  // calculating the indices that this thread will load into SMEM
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const uint strideA = numThreadsBlocktile / BK;
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const uint strideB = numThreadsBlocktile / BN;
  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  // register caches for As and Bs
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};
  // outer-most loop over block tiles
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
    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down
    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
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

// Wrapper function from your code
void runSgemm2DBlocktiling(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 && N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemm2DBlocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    // this is a hacky solution to the underlying problem
    // of not having proper bounds checking in the kernel
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemm2DBlocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}

// Sequential CPU matrix multiplication for verification
void cpuSgemm(int M, int N, int K, float alpha, const float *A, const float *B, 
              float beta, float *C) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = alpha * sum + beta * C[i * N + j];
    }
  }
}

// Verify results
bool verifyResults(const float *cpuC, const float *gpuC, int M, int N, float tolerance = 1e-5) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float diff = std::abs(cpuC[i * N + j] - gpuC[i * N + j]);
      if (diff > tolerance) {
        std::cout << "Mismatch at position (" << i << ", " << j << "): " 
                  << cpuC[i * N + j] << " vs " << gpuC[i * N + j] << std::endl;
        return false;
      }
    }
  }
  return true;
}

int main() {
  // Test with matrix dimensions that are multiples of block sizes
  const int M = 256;
  const int N = 256;
  const int K = 256;
  
  // Host memory
  float *h_A = new float[M * K];
  float *h_B = new float[K * N];
  float *h_C = new float[M * N];
  float *h_C_ref = new float[M * N];
  
  // Initialize matrices with random values
  for (int i = 0; i < M * K; ++i) {
    h_A[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < K * N; ++i) {
    h_B[i] = static_cast<float>(rand()) / RAND_MAX;
  }
  for (int i = 0; i < M * N; ++i) {
    h_C[i] = 0.0f;
    h_C_ref[i] = 0.0f;
  }
  
  // Device memory
  float *d_A, *d_B, *d_C;
  CHECK_CUDA_ERROR(cudaMalloc(&d_A, M * K * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_B, K * N * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc(&d_C, M * N * sizeof(float)));
  
  // Copy data to device
  CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(d_C, h_C, M * N * sizeof(float), cudaMemcpyHostToDevice));
  
  // Parameters for GEMM
  float alpha = 1.0f;
  float beta = 0.0f;
  
  // Create CUDA events for timing
  cudaEvent_t start, stop;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));
  
  // Start timing
  CHECK_CUDA_ERROR(cudaEventRecord(start));
  
  // Run our optimized GEMM
  runSgemm2DBlocktiling(M, N, K, alpha, d_A, d_B, beta, d_C);
  
  // Stop timing
  CHECK_CUDA_ERROR(cudaEventRecord(stop));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
  
  // Calculate elapsed time
  float milliseconds = 0;
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
  
  // Copy results back to host
  CHECK_CUDA_ERROR(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
  
  // Compute reference solution on CPU for verification
  cpuSgemm(M, N, K, alpha, h_A, h_B, beta, h_C_ref);
  
  // Verify results
  bool correct = verifyResults(h_C_ref, h_C, M, N);
  
  // Report results
  if (correct) {
    std::cout << "Test PASSED!" << std::endl;
    
    // Calculate and report performance
    float gflops = (2.0f * M * N * K) / (milliseconds * 1e6);
    std::cout << "Matrix dimensions: " << M << "x" << N << "x" << K << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    std::cout << "Execution time: " << milliseconds << " ms" << std::endl;
  } else {
    std::cout << "Test FAILED!" << std::endl;
  }
  
  // Clean up
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  delete[] h_C_ref;
  CHECK_CUDA_ERROR(cudaFree(d_A));
  CHECK_CUDA_ERROR(cudaFree(d_B));
  CHECK_CUDA_ERROR(cudaFree(d_C));
  CHECK_CUDA_ERROR(cudaEventDestroy(start));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop));
  
  return 0;
}
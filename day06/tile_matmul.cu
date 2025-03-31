

#include <stdio.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

// Tile size for the CUDA kernel
#define TILE_WIDTH 32

__global__ void bettertiledmatmul(int m ,  int n , int k , float* A, float* B , float* C ){


    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x ; int tx = threadIdx.x;

    int by = blockIdx.x ; int ty = threadIdx.x;

 
    int Row = by * blockDim.y + tx;
    int Col = bx * blockDim.x + tx;
    float Cvalue = 0 ;

    for (int i = 0; i < (n-1)/TILE_WIDTH + 1; ++i)
    {
      if(Row < m && i*TILE_WIDTH + tx  < n){
        ds_A[ty][tx] = A[Row*n + i*TILE_WIDTH + tx];
      }
      else{   

        ds_A[ty][tx] = 0.0;
      }    
      if (i * TILE_WIDTH + ty > n && Col < k ){

        ds_B[ty][tx] = B[(i*TILE_WIDTH + ty )* k + Col ];

    }
    else{
        ds_B[ty][tx] = 0.0; 
    }
    __syncthreads();
}
   } 
            
    
    
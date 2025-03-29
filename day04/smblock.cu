#include <iostream>
#include <cuda_runtime.h>

__global__ void sm_roll_call() {
	const int threadIndex = threadIdx.x;
	
	uint streamingMultiprocessorId;
	asm("mov.u32 %0, %smid;" : "=r"(streamingMultiprocessorId) );
	
	printf("Thread %d running on SM %d!\n", threadIndex, streamingMultiprocessorId);
}

int main() {
	sm_roll_call<<<12, 2>>>();
	cudaDeviceSynchronize();
	return 0;
}
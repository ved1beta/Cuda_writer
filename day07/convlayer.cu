#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for 2D convolution
__global__ void convolution2D(float* input, float* output, float* kernel,
                             int inputWidth, int inputHeight, 
                             int kernelSize, int outputWidth, int outputHeight) {
    // Calculate thread's position in output image
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if thread is within output image bounds
    if (x < outputWidth && y < outputHeight) {
        float sum = 0.0f;
        
        // Calculate half size of kernel (assuming kernel size is odd)
        int kernelRadius = kernelSize / 2;
        
        // Apply convolution
        for (int ky = 0; ky < kernelSize; ky++) {
            for (int kx = 0; kx < kernelSize; kx++) {
                // Calculate position in input image
                int inputX = x + kx - kernelRadius;
                int inputY = y + ky - kernelRadius;
                
                // Handle boundary conditions (zero padding)
                if (inputX >= 0 && inputX < inputWidth && 
                    inputY >= 0 && inputY < inputHeight) {
                    sum += input[inputY * inputWidth + inputX] * 
                           kernel[ky * kernelSize + kx];
                }
            }
        }
        
        // Write result to output
        output[y * outputWidth + x] = sum;
    }
}

// Host function to set up and launch the CUDA kernel
void launchConvolution(float* h_input, float* h_output, float* h_kernel,
                      int inputWidth, int inputHeight, int kernelSize) {
    // Calculate output dimensions
    int outputWidth = inputWidth;
    int outputHeight = inputHeight;
    
    // Allocate device memory
    float *d_input, *d_output, *d_kernel;
    size_t inputSize = inputWidth * inputHeight * sizeof(float);
    size_t outputSize = outputWidth * outputHeight * sizeof(float);
    size_t kernelSize2D = kernelSize * kernelSize * sizeof(float);
    
    cudaMalloc((void**)&d_input, inputSize);
    cudaMalloc((void**)&d_output, outputSize);
    cudaMalloc((void**)&d_kernel, kernelSize2D);
    
    // Copy data from host to device
    cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSize2D, cudaMemcpyHostToDevice);
    
    // Set up grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((outputWidth + blockDim.x - 1) / blockDim.x,
                 (outputHeight + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    convolution2D<<<gridDim, blockDim>>>(d_input, d_output, d_kernel,
                                        inputWidth, inputHeight,
                                        kernelSize, outputWidth, outputHeight);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }
    
    // Copy result back to host
    cudaMemcpy(h_output, d_output, outputSize, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

// Example usage
int main() {
    // Example image dimensions (32x32)
    const int width = 32;
    const int height = 32;
    
    // Example 3x3 kernel (Gaussian blur)
    const int kernelSize = 3;
    float kernel[kernelSize * kernelSize] = {
        1/16.0f, 2/16.0f, 1/16.0f,
        2/16.0f, 4/16.0f, 2/16.0f,
        1/16.0f, 2/16.0f, 1/16.0f
    };
    
    // Allocate memory for input and output
    float* input = (float*)malloc(width * height * sizeof(float));
    float* output = (float*)malloc(width * height * sizeof(float));
    
    // Initialize input with dummy data
    for (int i = 0; i < width * height; i++) {
        input[i] = (float)(i % 255);
    }
    
    // Launch the convolution
    launchConvolution(input, output, kernel, width, height, kernelSize);
    
    // Print a sample of the output (just for demonstration)
    printf("Sample output values:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%.2f ", output[i * width + j]);
        }
        printf("\n");
    }
    
    // Free host memory
    free(input);
    free(output);
    
    return 0;
}
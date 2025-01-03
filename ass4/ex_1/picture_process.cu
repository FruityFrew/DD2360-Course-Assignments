#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA_CALL(call)                                  \
    {                                                          \
        cudaError_t err = call;                                \
        if (err != cudaSuccess) {                              \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                      << " at line " << __LINE__ << std::endl; \
            return EXIT_FAILURE;                               \
        }                                                      \
    }

const int X = 800; // Width of the picture
const int Y = 600;  // Height of the picture

__global__ void PictureKernel(float* d_Pin, float* d_Pout, int n, int m) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if ((Row < m) && (Col < n)) {
        d_Pout[Row * n + Col] = 2 * d_Pin[Row * n + Col];
    }
}

int main() {
    const int size = X * Y * sizeof(float);
    float* h_Pin = new float[X * Y]; // Host input picture
    float* h_Pout = new float[X * Y]; // Host output picture

    // Initialize input data
    for (int i = 0; i < X * Y; i++) {
        h_Pin[i] = static_cast<float>(i % 256); // Example initialization
    }

    float *d_Pin, *d_Pout;

    // Allocate device memory
    CHECK_CUDA_CALL(cudaMalloc((void**)&d_Pin, size));
    CHECK_CUDA_CALL(cudaMalloc((void**)&d_Pout, size));

    // Copy data from host to device
    CHECK_CUDA_CALL(cudaMemcpy(d_Pin, h_Pin, size, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 blockDim(32, 16);
    dim3 gridDim((X + blockDim.x - 1) / blockDim.x, (Y + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    PictureKernel<<<gridDim, blockDim>>>(d_Pin, d_Pout, X, Y);
    CHECK_CUDA_CALL(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA_CALL(cudaMemcpy(h_Pout, d_Pout, size, cudaMemcpyDeviceToHost));

    // Verify results (optional)
    for (int i = 0; i < 10; i++) { // Print first 10 results
        std::cout << "h_Pout[" << i << "] = " << h_Pout[i] << std::endl;
    }

    // Free device memory
    cudaFree(d_Pin);
    cudaFree(d_Pout);

    // Free host memory
    delete[] h_Pin;
    delete[] h_Pout;

    return 0;
}
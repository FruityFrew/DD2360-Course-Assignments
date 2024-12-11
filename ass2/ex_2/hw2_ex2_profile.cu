#include <stdio.h>
#include <sys/time.h>
#include <random>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define DataType double

// Compute C = A * B
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns){
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < numARows && col < numBColumns) {
    DataType sum = 0;
    for (int k = 0; k < numAColumns; k++) {
      sum += A[row * numAColumns + k] * B[k * numBColumns + col];
    }
    C[row * numBColumns + col] = sum;
  }
}

int main(int argc, char **argv) {
  // Enable profiling
  cudaProfilerStart();
  
  DataType *hostA;
  DataType *hostB;
  DataType *hostC;
  DataType *resultRef;
  DataType *deviceA;
  DataType *deviceB;
  DataType *deviceC;
  int numARows = 1024;    // Fixed size for profiling
  int numAColumns = 1023;
  int numBRows = 1023;
  int numBColumns = 8193;
  int numCRows = numARows;
  int numCColumns = numBColumns;
  
  printf("Matrix dimensions: A(%d x %d) B(%d x %d) C(%d x %d)\n", 
         numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns);
  
  size_t sizeA = numARows * numAColumns * sizeof(DataType);
  size_t sizeB = numBRows * numBColumns * sizeof(DataType);
  size_t sizeC = numCRows * numCColumns * sizeof(DataType);

  // Allocate host memory
  hostA = (DataType*)malloc(sizeA);
  hostB = (DataType*)malloc(sizeB); 
  hostC = (DataType*)malloc(sizeC);
  resultRef = (DataType*)malloc(sizeC);

  // Initialize matrices with random values
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<DataType> dis(-1.0, 1.0);
  
  for (int i = 0; i < numARows * numAColumns; i++) {
    hostA[i] = dis(gen);
  }
  for (int i = 0; i < numBRows * numBColumns; i++) {
    hostB[i] = dis(gen);
  }
  
  // Calculate reference result on CPU
  for (int i = 0; i < numCRows; i++) {
    for (int j = 0; j < numCColumns; j++) {
      DataType sum = 0;
      for (int k = 0; k < numAColumns; k++) {
        sum += hostA[i * numAColumns + k] * hostB[k * numBColumns + j];
      }
      resultRef[i * numCColumns + j] = sum;
    }
  }

  // Allocate device memory
  cudaMalloc(&deviceA, sizeA);
  cudaMalloc(&deviceB, sizeB);
  cudaMalloc(&deviceC, sizeC);

  // Copy data to device
  cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);

  // Set up timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Configure kernel launch parameters
  dim3 blockDim(16, 16);  // 256 threads per block
  dim3 gridDim(
    (numCColumns + blockDim.x - 1) / blockDim.x,  // 4 blocks
    (numCRows + blockDim.y - 1) / blockDim.y      // 4 blocks
  );

  printf("Grid dimensions: (%d, %d)\n", gridDim.x, gridDim.y);
  printf("Block dimensions: (%d, %d)\n", blockDim.x, blockDim.y);
  printf("Total threads: %d\n", gridDim.x * gridDim.y * blockDim.x * blockDim.y);

  // Warmup run
  gemm<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
  cudaDeviceSynchronize();

  // Timed run with profiling
  cudaEventRecord(start);
  gemm<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns);
  cudaEventRecord(stop);
  
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  
  printf("Kernel execution time: %f ms\n", milliseconds);

  // Copy result back to host
  cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);

  // Verify results
  bool correct = true;
  double epsilon = 1e-8;
  for (int i = 0; i < numCRows * numCColumns; i++) {
    if (abs(hostC[i] - resultRef[i]) > epsilon) {
      correct = false;
      printf("Mismatch at index %d: GPU = %f, CPU = %f\n", 
             i, hostC[i], resultRef[i]);
      break;
    }
  }
  printf("Results %s\n", correct ? "MATCH" : "DO NOT MATCH");

  // Cleanup
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  free(hostA);
  free(hostB);
  free(hostC);
  free(resultRef);

  // Stop profiling
  cudaProfilerStop();
  
  return 0;
}
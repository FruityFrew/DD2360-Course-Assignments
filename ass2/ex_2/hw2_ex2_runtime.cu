#include <stdio.h>
#include <sys/time.h>
#include <random>
#include <cuda_runtime.h>
#include <vector>
#include <string>

// Configurable DataType
#define DataType float  // Change to double for second experiment

// Timer class for CPU timing
class Timer {
    struct timeval start_time, end_time;
public:
    void start() { gettimeofday(&start_time, NULL); }
    double end() {
        gettimeofday(&end_time, NULL);
        return (end_time.tv_sec - start_time.tv_sec) * 1000.0 + 
               (end_time.tv_usec - start_time.tv_usec) / 1000.0; // Returns ms
    }
};

// Kernel remains the same
__global__ void gemm(DataType *A, DataType *B, DataType *C, int numARows,
                      int numAColumns, int numBRows, int numBColumns) {
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

struct TimingResult {
    double h2d_time;
    double kernel_time;
    double d2h_time;
    int matrix_size;
};

TimingResult benchmark_size(int size) {
    int numARows = size;
    int numAColumns = size;
    int numBRows = size;
    int numBColumns = size;
    
    size_t sizeA = numARows * numAColumns * sizeof(DataType);
    size_t sizeB = numBRows * numBColumns * sizeof(DataType);
    size_t sizeC = numARows * numBColumns * sizeof(DataType);

    // Allocate host memory
    DataType *hostA = (DataType*)malloc(sizeA);
    DataType *hostB = (DataType*)malloc(sizeB);
    DataType *hostC = (DataType*)malloc(sizeC);

    // Initialize matrices
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<DataType> dis(-1.0, 1.0);
    
    for (int i = 0; i < numARows * numAColumns; i++) hostA[i] = dis(gen);
    for (int i = 0; i < numBRows * numBColumns; i++) hostB[i] = dis(gen);

    // Allocate device memory
    DataType *deviceA, *deviceB, *deviceC;
    cudaMalloc(&deviceA, sizeA);
    cudaMalloc(&deviceB, sizeB);
    cudaMalloc(&deviceC, sizeC);

    // Timing measurements
    Timer timer;
    TimingResult result;
    result.matrix_size = size;

    // Measure H2D time
    timer.start();
    cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);
    result.h2d_time = timer.end();

    // Configure and launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim(
        (numBColumns + blockDim.x - 1) / blockDim.x,
        (numARows + blockDim.y - 1) / blockDim.y
    );

    // Measure kernel time
    cudaDeviceSynchronize();
    timer.start();
    gemm<<<gridDim, blockDim>>>(deviceA, deviceB, deviceC, 
                                numARows, numAColumns, numBRows, numBColumns);
    cudaDeviceSynchronize();
    result.kernel_time = timer.end();

    // Measure D2H time
    timer.start();
    cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);
    result.d2h_time = timer.end();

    // Cleanup
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    free(hostA);
    free(hostB);
    free(hostC);

    return result;
}

int main() {
    // Test different matrix sizes
    std::vector<int> sizes = {1024, 2048, 4096, 8192};
    std::vector<TimingResult> results;

    for (int size : sizes) {
        printf("Testing size %d x %d\n", size, size);
        TimingResult result = benchmark_size(size);
        results.push_back(result);
        printf("H2D: %.2f ms, Kernel: %.2f ms, D2H: %.2f ms, Total: %.2f ms\n",
               result.h2d_time, result.kernel_time, result.d2h_time,
               result.h2d_time + result.kernel_time + result.d2h_time);
    }

    // Generate data for plotting
    printf("\nData for plotting:\n");
    printf("size,h2d,kernel,d2h\n");
    for (const auto& r : results) {
        printf("%d,%.2f,%.2f,%.2f\n", 
               r.matrix_size, r.h2d_time, r.kernel_time, r.d2h_time);
    }

    return 0;
}
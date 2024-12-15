#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <random>
#include <cuda_runtime.h>

#define NUM_BINS 4096
#define BIN_MAX 127

__global__ void histogram_kernel(unsigned int *input, unsigned int *bins,
                                 unsigned int num_elements,
                                 unsigned int num_bins) {
    // Use shared memory for the histogram bins
    extern __shared__ unsigned int shared_bins[];

    // Initialize shared memory bins to zero
    int tid = threadIdx.x;
    for (int i = tid; i < num_bins; i += blockDim.x) {
        shared_bins[i] = 0;
    }
    __syncthreads();

    // Compute the histogram using shared memory and atomics
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (idx < num_elements) {
        atomicAdd(&shared_bins[input[idx]], 1);
        idx += stride;
     }

    __syncthreads();

    // Merge shared memory bins into global memory bins using atomics
    for (int i = tid; i < num_bins; i += blockDim.x) {
        atomicAdd(&bins[i], shared_bins[i]);
    }
}

__global__ void convert_kernel(unsigned int *bins, unsigned int num_bins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    while (idx < num_bins) {
        if (bins[idx] > BIN_MAX) {
            bins[idx] = BIN_MAX;
        }
        idx += stride;
    }
}

int main(int argc, char **argv) {
    int inputLength = 1024000;
    printf("The input length is %d\n", inputLength);

    unsigned int *hostInput, *hostBins, *resultRef;
    unsigned int *deviceInput, *deviceBins;

    // Allocate Host memory for input and output
    hostInput = (unsigned int *)malloc(inputLength * sizeof(unsigned int));
    hostBins = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));
    resultRef = (unsigned int *)malloc(NUM_BINS * sizeof(unsigned int));

    // Initialize hostInput to random numbers whose values range from 0 to (NUM_BINS - 1)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, NUM_BINS - 1);
    for (int i = 0; i < inputLength; ++i) {
        hostInput[i] = dis(gen);
    }

    // Create reference result in CPU
    for (int i = 0; i < NUM_BINS; ++i) {
        resultRef[i] = 0;
    }
    for (int i = 0; i < inputLength; ++i) {
        if (resultRef[hostInput[i]] < BIN_MAX) {
            resultRef[hostInput[i]]++;
        }
    }

    // Allocate GPU memory
    cudaMalloc(&deviceInput, inputLength * sizeof(unsigned int));
    cudaMalloc(&deviceBins, NUM_BINS * sizeof(unsigned int));

    // Copy memory to the GPU
    cudaMemcpy(deviceInput, hostInput, inputLength * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(deviceBins, 0, NUM_BINS * sizeof(unsigned int));

    // Initialize grid and block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (inputLength + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the histogram kernel
    size_t sharedMemSize = NUM_BINS * sizeof(unsigned int); 
    histogram_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(deviceInput, deviceBins, inputLength, NUM_BINS);

    // Launch the cleanup kernel
    blocksPerGrid = (NUM_BINS + threadsPerBlock - 1) / threadsPerBlock;
    convert_kernel<<<blocksPerGrid, threadsPerBlock>>>(deviceBins, NUM_BINS);

    // Copy the GPU memory back to the CPU
    cudaMemcpy(hostBins, deviceBins, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Compare the output with the reference
    bool match = true;
    for (int i = 0; i < NUM_BINS; ++i) {
        if (hostBins[i] != resultRef[i]) {
            printf("Mismatch at bin %d: GPU %d, CPU %d\n", i, hostBins[i], resultRef[i]);
            match = false;
            break;
        }
    }

    if (match) {
        printf("Results match!\n");
    } else {
        printf("Results do not match!\n");
    }

    // Free the GPU memory
    cudaFree(deviceInput);
    cudaFree(deviceBins);

    // Free the CPU memory
    free(hostInput);
    free(hostBins);
    free(resultRef);

    return 0;
}

#include <stdio.h>
#include <sys/time.h>

typedef float DataType;

// Kernel for non-streamed version
__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        out[idx] = in1[idx] + in2[idx];
    }
}

double getTimeMs() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((double)tv.tv_sec * 1000 + (double)tv.tv_usec / 1000);
}

// Function to run non-streamed version
double runNonStreamed(int inputLength) {
    DataType *hostInput1, *hostInput2, *hostOutput;
    DataType *deviceInput1, *deviceInput2, *deviceOutput;
    
    // Allocate host memory
    cudaMallocHost(&hostInput1, inputLength * sizeof(DataType));
    cudaMallocHost(&hostInput2, inputLength * sizeof(DataType));
    cudaMallocHost(&hostOutput, inputLength * sizeof(DataType));
    
    // Initialize input vectors
    for (int i = 0; i < inputLength; i++) {
        hostInput1[i] = rand() / (DataType)RAND_MAX;
        hostInput2[i] = rand() / (DataType)RAND_MAX;
    }
    
    // Allocate GPU memory
    cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
    cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
    cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));
    
    double startTime = getTimeMs();
    
    // Copy inputs to device
    cudaMemcpy(deviceInput1, hostInput1, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (inputLength + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
    
    // Copy result back to host
    cudaMemcpy(hostOutput, deviceOutput, inputLength * sizeof(DataType), cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
    double endTime = getTimeMs();
    
    // Cleanup
    cudaFreeHost(hostInput1);
    cudaFreeHost(hostInput2);
    cudaFreeHost(hostOutput);
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);
    
    return endTime - startTime;
}

// Function to run streamed version
double runStreamed(int inputLength, int S_seg) {
    DataType *hostInput1, *hostInput2, *hostOutput;
    DataType *deviceInput1, *deviceInput2, *deviceOutput;
    
    cudaMallocHost(&hostInput1, inputLength * sizeof(DataType));
    cudaMallocHost(&hostInput2, inputLength * sizeof(DataType));
    cudaMallocHost(&hostOutput, inputLength * sizeof(DataType));
    
    for (int i = 0; i < inputLength; i++) {
        hostInput1[i] = rand() / (DataType)RAND_MAX;
        hostInput2[i] = rand() / (DataType)RAND_MAX;
    }
    
    cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
    cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
    cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));
    
    cudaStream_t streams[4];
    for (int i = 0; i < 4; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    double startTime = getTimeMs();
    
    const int threadsPerBlock = 256;
    for (int offset = 0; offset < inputLength; offset += S_seg) {
        int currentSegSize = min(S_seg, inputLength - offset);
        int streamIdx = (offset/S_seg) % 4;
        int blocksPerGrid = (currentSegSize + threadsPerBlock - 1) / threadsPerBlock;
        
        cudaMemcpyAsync(&deviceInput1[offset], &hostInput1[offset],
                        currentSegSize * sizeof(DataType), 
                        cudaMemcpyHostToDevice, streams[streamIdx]);
        cudaMemcpyAsync(&deviceInput2[offset], &hostInput2[offset],
                        currentSegSize * sizeof(DataType), 
                        cudaMemcpyHostToDevice, streams[streamIdx]);
        
        vecAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[streamIdx]>>>
              (&deviceInput1[offset], &deviceInput2[offset], 
               &deviceOutput[offset], currentSegSize);
        
        cudaMemcpyAsync(&hostOutput[offset], &deviceOutput[offset],
                        currentSegSize * sizeof(DataType), 
                        cudaMemcpyDeviceToHost, streams[streamIdx]);
    }
    
    cudaDeviceSynchronize();
    double endTime = getTimeMs();
    
    for (int i = 0; i < 4; i++) {
        cudaStreamDestroy(streams[i]);
    }
    
    cudaFreeHost(hostInput1);
    cudaFreeHost(hostInput2);
    cudaFreeHost(hostOutput);
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);
    
    return endTime - startTime;
}

int main() {
    // Test different vector lengths
    int vectorLengths[] = {1024*1024, 2*1024*1024, 4*1024*1024, 8*1024*1024};
    
    printf("Vector adding...\n");
    
    for (int i = 0; i < 4; i++) {
        int length = vectorLengths[i];
        printf("Non_Streamed, Streamed, Vector_length = %d\n",  length);
        double nonStreamedTime = runNonStreamed(length);
        for (int j = 0; j < 9; j++) {
            double streamedTime = runStreamed(length, length/(1 << j)); // Use length/2^j as segment size
            printf("%.3f, %.3f, S_seg = %d\n", nonStreamedTime, streamedTime, length/(1 << j));
        }
        // double streamedTime = runStreamed(length, length/8); // Use length/8 as segment size
        
        // printf("%d,%.3f,%.3f\n", length, nonStreamedTime, streamedTime);
    }
    // int length = vectorLengths[0];
    //     printf("Non_Streamed, Streamed, Vector_length = %d\n",  length);
    //     double nonStreamedTime = runNonStreamed(length);
    //     for (int j = 0; j < 9; j++) {
    //         double streamedTime = runStreamed(length, length/(1 << j)); // Use length/2^j as segment size
    //         printf("%.3f, %.3f, S_seg = %d\n", nonStreamedTime, streamedTime, length/(1 << j));
    //     }
    return 0;
}
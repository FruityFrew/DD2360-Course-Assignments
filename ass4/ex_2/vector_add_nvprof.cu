#include <stdio.h>
#include <sys/time.h>

typedef float DataType;

__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        out[idx] = in1[idx] + in2[idx];
    }
}

int main(int argc, char **argv) {
    int inputLength = (argc > 1) ? atoi(argv[1]) : 1048576;
    int S_seg = inputLength / 4;  // Segment size

    // Allocate pinned memory
    DataType *hostInput1, *hostInput2, *hostOutput;
    cudaMallocHost(&hostInput1, inputLength * sizeof(DataType));
    cudaMallocHost(&hostInput2, inputLength * sizeof(DataType));
    cudaMallocHost(&hostOutput, inputLength * sizeof(DataType));

    // Initialize input vectors
    for (int i = 0; i < inputLength; i++) {
        hostInput1[i] = rand() / (DataType)RAND_MAX;
        hostInput2[i] = rand() / (DataType)RAND_MAX;
    }

    // Allocate GPU memory
    DataType *deviceInput1, *deviceInput2, *deviceOutput;
    cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
    cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
    cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));

    // Create streams and events
    cudaStream_t streams[4];
    cudaEvent_t startEvent, stopEvent;
    cudaEvent_t copyCompleteEvents[4];
    cudaEvent_t kernelCompleteEvents[4];
    
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    
    for (int i = 0; i < 4; i++) {
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&copyCompleteEvents[i]);
        cudaEventCreate(&kernelCompleteEvents[i]);
    }

    // Record start event
    cudaEventRecord(startEvent);

    // Process segments using multiple streams
    const int threadsPerBlock = 256;
    
    for (int offset = 0; offset < inputLength; offset += S_seg) {
        int currentSegSize = min(S_seg, inputLength - offset);
        int streamIdx = (offset/S_seg) % 4;
        int blocksPerGrid = (currentSegSize + threadsPerBlock - 1) / threadsPerBlock;

        // Asynchronous memory copies to device
        cudaMemcpyAsync(&deviceInput1[offset], 
                       &hostInput1[offset],
                       currentSegSize * sizeof(DataType), 
                       cudaMemcpyHostToDevice, 
                       streams[streamIdx]);
                       
        cudaMemcpyAsync(&deviceInput2[offset], 
                       &hostInput2[offset],
                       currentSegSize * sizeof(DataType), 
                       cudaMemcpyHostToDevice, 
                       streams[streamIdx]);
                       
        // Record completion of copies
        cudaEventRecord(copyCompleteEvents[streamIdx], streams[streamIdx]);

        // Launch kernel
        vecAdd<<<blocksPerGrid, threadsPerBlock, 0, streams[streamIdx]>>>
              (&deviceInput1[offset], 
               &deviceInput2[offset], 
               &deviceOutput[offset], 
               currentSegSize);
               
        // Record completion of kernel
        cudaEventRecord(kernelCompleteEvents[streamIdx], streams[streamIdx]);

        // Asynchronous memory copy back to host
        cudaMemcpyAsync(&hostOutput[offset], 
                       &deviceOutput[offset],
                       currentSegSize * sizeof(DataType), 
                       cudaMemcpyDeviceToHost, 
                       streams[streamIdx]);
    }

    // Record stop event
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    // Calculate and print elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
    printf("Execution time: %f ms\n", milliseconds);

    // Cleanup
    for (int i = 0; i < 4; i++) {
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(copyCompleteEvents[i]);
        cudaEventDestroy(kernelCompleteEvents[i]);
    }
    
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    cudaFreeHost(hostInput1);
    cudaFreeHost(hostInput2);
    cudaFreeHost(hostOutput);
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    return 0;
}
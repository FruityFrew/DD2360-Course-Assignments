#include <iostream>
#include <chrono>
#include <random>
#include <cassert>
#include <fstream>

#include <cuda_runtime.h>

#ifndef DEBUG_ENABLED
#define DEBUG_ENABLED false
#endif

using DataType = double;


__global__ void vecAdd(DataType *in1, DataType *in2, DataType *out, int len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        out[idx] = in1[idx] + in2[idx];
        // printf("out[%d] = %f\n", idx, out[idx]);
    }
}

//@@ Insert code to implement timer start
//@@ Insert code to implement timer stop
class Timer
{
public:
    void start()
    {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void stop()
    {
        end_time = std::chrono::high_resolution_clock::now();
    }

    double durr() const
    {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
        return duration;
    }

private:
    std::chrono::high_resolution_clock::time_point start_time, end_time;
};



int main(int argc, char **argv)
{

    //@@ Insert code below to read in inputLength from args
    if (argc != 2) {
        std::cerr << "Wrong argument count error" << std::endl;
        std::cout << "Usage: " << argv[0] << " <input_length>" << std::endl;
        return 1;   // error indication
    }

    int inputLength = std::stoi(argv[1]);

    if (DEBUG_ENABLED) std::cout << "Input length: " << inputLength << std::endl;

    //@@ Insert code below to allocate Host memory for input and output
    std::vector<DataType> hostInput1(inputLength);
    std::vector<DataType> hostInput2(inputLength);

    std::vector<DataType> hostOutput(inputLength);
    std::vector<DataType> resultRef(inputLength);
    
    Timer sum_on_device_timer, copy_to_device_timer, copy_from_device_timer;

    //@@ Insert code below to initialize hostInput1 and hostInput2 to random numbers, and create reference result in CPU
    std::random_device rng_dev;
    std::mt19937 rng(rng_dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0,1e8); // distribution in range [1, 6]

    for (int i = 0; i < inputLength; ++i) {
        auto random_num = static_cast<double>(dist(rng));
        hostInput1[i] = random_num;

        random_num = static_cast<double>(dist(rng));
        hostInput2[i] = random_num;

        resultRef[i] = hostInput1[i] + hostInput2[i];
        if (DEBUG_ENABLED) std::cout << hostInput1[i] << " + " << hostInput2[i] << " = " << resultRef[i] << std::endl;
    }

    //@@ Insert code below to allocate GPU memory here
    DataType *deviceInput1, *deviceInput2, *deviceOutput;
    cudaMalloc(&deviceInput1, inputLength * sizeof(DataType));
    cudaMalloc(&deviceInput2, inputLength * sizeof(DataType));
    cudaMalloc(&deviceOutput, inputLength * sizeof(DataType));

    //@@ Insert code to below to Copy memory to the GPU here
    copy_to_device_timer.start();
    cudaMemcpy(deviceInput1, hostInput1.data(), inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2.data(), inputLength * sizeof(DataType), cudaMemcpyHostToDevice);
    copy_to_device_timer.stop();


    //@@ Initialize the 1D grid and block dimensions here
    int threadsPerBlock = 256;
    int blocksPerGrid = (inputLength + threadsPerBlock - 1) / threadsPerBlock;

    //@@ Launch the GPU Kernel here
    sum_on_device_timer.start();

    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
    // // ERROR DEBUGGING START
    // cudaError_t kernelErr = cudaGetLastError();
    // if (kernelErr != cudaSuccess) {
    //     std::cerr << "Kernel launch error: " << cudaGetErrorString(kernelErr) << std::endl;
    //     return -1;
    // }
    // // ERROR DEBUGGING END

    cudaDeviceSynchronize();

    sum_on_device_timer.stop();

    //@@ Copy the GPU memory back to the CPU here
    copy_from_device_timer.start();
    cudaMemcpy(hostOutput.data(), deviceOutput, inputLength * sizeof(DataType), cudaMemcpyDeviceToHost);
    copy_from_device_timer.stop();

    //@@ Insert code below to compare the output with the reference
    int match_count = 0;
    for (int i = 0; i < inputLength; ++i) {
        // assert(hostOutput[i] == resultRef[i]);
        match_count += (hostOutput[i] == resultRef[i]) ? 1 : 0;
        if (DEBUG_ENABLED) {
            std::cout << "calc val:" << hostOutput[i] 
                      << ", ref val=" << resultRef[i] 
                      << ", delta=" << std::abs(resultRef[i] - hostOutput[i]) 
                      << std::endl;
        }
    }
    std::cout << "Matched " << match_count << " out of " << inputLength << std::endl;
    assert(match_count == inputLength);

    //@@ Free the GPU memory here
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    //@@ Free the CPU memory here
    // no need for explcit destructor call; vectors memory will be freed at the end of a scope it belongs to

    std::cout << "Time to copy to device:   " << copy_to_device_timer.durr()   << "μs" << std::endl;
    std::cout << "Time to add vectors:      " << sum_on_device_timer.durr()    << "μs" << std::endl;
    std::cout << "TIme to copy from device: " << copy_from_device_timer.durr() << "μs" << std::endl;

    std::ofstream log_file;
    log_file.open("logged_runs_v2.txt", std::ios_base::app);
    log_file << inputLength << "," 
             << copy_to_device_timer.durr() << ","
             << copy_from_device_timer.durr() << "," 
             << sum_on_device_timer.durr() << "\n";
    log_file.close();

    return 0;
}

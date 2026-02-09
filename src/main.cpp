#include <iostream>
#include <cstdlib>

#include <cuda_runtime.h>
#include <NvInfer.h>

int main(int /*argc*/, char* /*argv*/[]) {
    std::cout << "=== VibeVoice Windows API Server ===" << std::endl;

    // CUDA version
    int cuda_runtime_ver = 0;
    cudaRuntimeGetVersion(&cuda_runtime_ver);
    int cuda_major = cuda_runtime_ver / 1000;
    int cuda_minor = (cuda_runtime_ver % 1000) / 10;
    std::cout << "CUDA Runtime: " << cuda_major << "." << cuda_minor << std::endl;

    // CUDA device info
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    std::cout << "CUDA Devices:  " << device_count << std::endl;
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "  [" << i << "] " << prop.name
                  << " (SM " << prop.major << "." << prop.minor
                  << ", " << (prop.totalGlobalMem / (1024 * 1024)) << " MB)" << std::endl;
    }

    // TensorRT version (encoding: major*1000 + minor*100 + patch)
    int trt_ver = getInferLibVersion();
    int trt_major = trt_ver / 1000;
    int trt_minor = (trt_ver % 1000) / 100;
    int trt_patch = trt_ver % 100;
    std::cout << "TensorRT:      " << trt_major << "." << trt_minor << "." << trt_patch << std::endl;

    std::cout << "\nBuild test passed. All libraries linked successfully." << std::endl;
    return 0;
}

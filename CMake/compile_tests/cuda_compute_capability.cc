#include <cuda.h>

#include <iostream>

int main() {
  // cudaDeviceProp device_properties;
  // const cudaError_t error = cudaGetDeviceProperties(&device_properties, /*device*/ 0);
  CUresult result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    std::cout << "[cuInit] CUDA error: " << result << '\n';
    return -1;
  }

  int numTotalDevice = 0;
  result = cuDeviceGetCount(&numTotalDevice);
  if (result != CUDA_SUCCESS) {
    std::cout << "[cuDeviceGetCount] CUDA error: " << result << '\n';
    return -1;
  } else if (numTotalDevice == 0) {
    std::cout << "No cuda device detected" << '\n';
    return -1;
  }

  for (int i = 0; i < numTotalDevice; ++i) {
    CUdevice dev;
    CUcontext ctx;
    cuDeviceGet(&dev, i);
#if 0
    result = cuDevicePrimaryCtxRetain(&ctx, dev);
    if (result != CUDA_SUCCESS) {
      std::cout << "[cuDevicePrimaryCtxRetain] CUDA error: " << result << '\n';
      return -1;
    }
    result = cuCtxSetCurrent(ctx);
    if (result != CUDA_SUCCESS) {
      std::cout << "[cuCtxSetCurrent] CUDA error: " << result << '\n';
      return -1;
    }
#endif

    int major, minor;
    result = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    if (result != CUDA_SUCCESS) {
      std::cout << "[cuDeviceGetAttribute: minor] CUDA error: " << result << '\n';
      return -1;
    }
    result = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    if (result != CUDA_SUCCESS) {
      std::cout << "[cuDeviceGetAttribute: major] CUDA error: " << result << '\n';
      return -1;
    }
    unsigned int const compute_capability = major * 10 + minor;
    std::cout << compute_capability << ' ';
  }

  return 0;
}
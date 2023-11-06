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
  int major, minor;
  result = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, (CUdevice)0);
  if (result != CUDA_SUCCESS) {
    std::cout << "[cuDeviceGetAttribute: minor] CUDA error: " << result << '\n';
    return -1;
  }
  result = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, (CUdevice)0);
  if (result != CUDA_SUCCESS) {
    std::cout << "[cuDeviceGetAttribute: major] CUDA error: " << result << '\n';
    return -1;
  }
  unsigned int const compute_capability = major * 10 + minor;
  std::cout << compute_capability;

  return 0;
}
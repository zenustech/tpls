#include <musa.h>
#include <musa_runtime.h>

#include <iostream>

int main() {
  MUresult result = muInit(0);
  if (result != MUSA_SUCCESS) {
    std::cout << "[muInit] MUSA error: " << result << '\n';
    return -1;
  }

  int numTotalDevice = 0;
  result = muDeviceGetCount(&numTotalDevice);
  if (result != MUSA_SUCCESS) {
    std::cout << "[muDeviceGetCount] MUSA error: " << result << '\n';
    return -1;
  } else if (numTotalDevice == 0) {
    std::cout << "No musa device detected" << '\n';
    return -1;
  }

  for (int i = 0; i < numTotalDevice; ++i) {
    MUdevice dev = 0;
    MUcontext ctx;
    muDeviceGet(&dev, i);

    int major, minor;
    result = muDeviceGetAttribute(&minor, MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    if (result != MUSA_SUCCESS) {
      std::cout << "[muDeviceGetAttribute: minor] MUSA error: " << result << '\n';
      return -1;
    }
    result = muDeviceGetAttribute(&major, MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    if (result != MUSA_SUCCESS) {
      std::cout << "[muDeviceGetAttribute: major] MUSA error: " << result << '\n';
      return -1;
    }
    unsigned int const compute_capability = major * 10 + minor;
    std::cout << compute_capability << ' ';
  }

  return 0;
}


#include <iostream>

#include "vulkan/vulkan_core.h"

int main() {
  if (VK_HEADER_VERSION_COMPLETE >= VK_MAKE_API_VERSION(0, 1, 4, 0)) {
    std::cout << 1 << std::endl;
  } else {
    std::cout << 0 << std::endl;
  }
  return 0;
}
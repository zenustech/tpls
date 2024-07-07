
cmake_minimum_required(VERSION 3.17.0)
project(zswhereami LANGUAGES CXX)

set(WHEREAMI_LIBRARY_TYPE STATIC)

add_library(zswhereami ${WHEREAMI_LIBRARY_TYPE} ${PROJECT_SOURCE_DIR}/whereami.c)
set_target_properties(zswhereami PROPERTIES LINKER_LANGUAGE CXX)
# target_compile_features(zswhereami PRIVATE cxx_std_14)
if (${CMAKE_CXX_COMPILER_ID} MATCHES ".*Clang")
  target_compile_options(zswhereami PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>: -std=c++14>
  )
else()
  target_compile_features(zswhereami PRIVATE cxx_std_14)
endif()
set_target_properties(zswhereami PROPERTIES OUTPUT_NAME zswhereami POSITION_INDEPENDENT_CODE ON)
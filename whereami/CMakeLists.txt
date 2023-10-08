
cmake_minimum_required(VERSION 3.17.0)
project(zswhereami LANGUAGES CXX)

if(WHEREAMI_BUILD_SHARED_LIBS)
    set(WHEREAMI_LIBRARY_TYPE SHARED)
else()
    set(WHEREAMI_LIBRARY_TYPE STATIC)
endif()

add_library(zswhereami ${WHEREAMI_LIBRARY_TYPE} ${PROJECT_SOURCE_DIR}/whereami.c)
set_target_properties(zswhereami PROPERTIES LINKER_LANGUAGE CXX)
target_compile_features(zswhereami PRIVATE cxx_std_14)
set_target_properties(zswhereami PROPERTIES OUTPUT_NAME zswhereami POSITION_INDEPENDENT_CODE ON)
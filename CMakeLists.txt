cmake_minimum_required(VERSION 3.18)

option(ZS_ENABLE_INSTALL "Enable install rule" OFF)
option(ZS_ENABLE_OPENGL "Enable opengl" OFF)
option(ZS_ENABLE_OPENVDB "Enable openvdb" OFF)
option(ZS_ENABLE_CUDA "Enable cuda" OFF)
option(ZS_ENABLE_OPENMP "Enable openmp" OFF)
option(ZS_ENABLE_PTHREADS "Enable pthreads" OFF)
option(ZS_ENABLE_PARTIO "Enable partio" ON)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake ${CMAKE_CURRENT_SOURCE_DIR}/CMake ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/CMake ${CMAKE_CURRENT_SOURCE_DIR}/CMake ${CMAKE_CURRENT_BINARY_DIR})

add_library(zsproj_deps INTERFACE)

# opengl
if (ZS_ENABLE_OPENGL)
    find_package(OpenGL)
    if (TARGET OpenGL::GL)
        target_link_libraries(zsproj_deps INTERFACE OpenGL::GL)
    else()
        set(ZS_ENABLE_OPENGL OFF)
        # message(FATAL_ERROR "OpenGL not found!")
    endif()
endif(ZS_ENABLE_OPENGL)
set(ZS_ENABLE_OPENGL ${ZS_ENABLE_OPENGL} PARENT_SCOPE)

# cuda
if (ZS_ENABLE_CUDA)
    add_library(zpc_cuda_deps INTERFACE)
    target_compile_features(zpc_cuda_deps INTERFACE cuda_std_17)
    find_package(CUDAToolkit)
    if (TARGET CUDA::toolkit)
        target_link_libraries(zpc_cuda_deps INTERFACE CUDA::cudart CUDA::cuda_driver CUDA::cublas CUDA::cusparse CUDA::cusolver CUDA::toolkit CUDA::nvrtc)
    else()
        set(ZS_ENABLE_CUDA OFF)
    endif()
endif(ZS_ENABLE_CUDA)
set(ZS_ENABLE_CUDA ${ZS_ENABLE_CUDA} PARENT_SCOPE)

# threads
if (ZS_ENABLE_PTHREADS)
  find_package(Threads)
  if (NOT TARGET Threads::Threads)
    set(ZS_ENABLE_PTHREADS OFF)
  endif()
  target_link_libraries(zsproj_deps INTERFACE Threads::Threads)
endif(ZS_ENABLE_PTHREADS)
set(ZS_ENABLE_PTHREADS ${ZS_ENABLE_PTHREADS} PARENT_SCOPE)

# openmp
if (ZS_ENABLE_OPENMP)
    add_library(zpc_omp_deps INTERFACE)
    find_package(OpenMP)
    if (TARGET OpenMP::OpenMP_CXX)
        # https://gitlab.kitware.com/cmake/cmake/-/issues/17256
        # this target already imported in zen
        target_link_libraries(zpc_omp_deps INTERFACE OpenMP::OpenMP_CXX)
    else()
        set(ZS_ENABLE_OPENMP OFF)
        # message(FATAL_ERROR "OpenMP not found!")
    endif()
endif(ZS_ENABLE_OPENMP)
set(ZS_ENABLE_OPENMP ${ZS_ENABLE_OPENMP} PARENT_SCOPE)

# openvdb
if (ZS_ENABLE_OPENVDB)
    add_library(zsopenvdb INTERFACE)
    find_package(OpenVDB)
    if (TARGET OpenVDB::openvdb)
        target_link_libraries(zsopenvdb INTERFACE OpenVDB::openvdb)
    else()
        set(ZS_ENABLE_OPENVDB OFF)
    endif()
endif(ZS_ENABLE_OPENVDB)
set(ZS_ENABLE_OPENVDB ${ZS_ENABLE_OPENVDB} PARENT_SCOPE)

# tpl that needs compilation
# partio
if (ZS_ENABLE_PARTIO)
    add_subdirectory(partio EXCLUDE_FROM_ALL)
endif(ZS_ENABLE_PARTIO)
set(ZS_ENABLE_PARTIO ${ZS_ENABLE_PARTIO} PARENT_SCOPE)
# loguru
add_subdirectory(loguru EXCLUDE_FROM_ALL)
add_dependencies(zsproj_deps zsloguru)
target_link_libraries(zsproj_deps INTERFACE zsloguru)

# other local tpls
# fmt
target_compile_definitions(zsproj_deps INTERFACE FMT_HEADER_ONLY)
# Catch2
# gcem
# function_ref
# jitify
# rapidjson
# cxxopts
# magic_enum
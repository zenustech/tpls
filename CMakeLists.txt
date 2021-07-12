cmake_minimum_required(VERSION 3.18)

option(ZS_ENABLE_INSTALL "Enable install rule" OFF)
option(ZS_ENABLE_OPENGL "Enable opengl" OFF)
option(ZS_ENABLE_OPENVDB "Enable openvdb" OFF)
option(ZS_ENABLE_CUDA "Enable cuda" ON)
option(ZS_ENABLE_OPENMP "Enable openmp" ON)
option(ZS_ENABLE_PTHREADS "Enable pthreads" ON)
option(ZS_ENABLE_PARTIO "Enable partio" ON)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake ${CMAKE_CURRENT_SOURCE_DIR}/CMake ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/CMake ${CMAKE_CURRENT_SOURCE_DIR}/CMake ${CMAKE_CURRENT_BINARY_DIR})

# add_library(zsproj_deps INTERFACE IMPORTED)

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

# openvdb
if (ZS_ENABLE_OPENVDB)
    add_library(zsopenvdb INTERFACE)
    find_package(OpenVDB)
    if (TARGET OpenVDB::openvdb)
        target_link_libraries(zsopenvdb INTERFACE OpenVDB::openvdb)
    else()
        include(CMake/conan.cmake)
        conan_cmake_configure(REQUIRES openvdb/8.0.1
                            GENERATORS cmake_find_package)
        conan_cmake_autodetect(settings)
        conan_cmake_install(PATH_OR_REFERENCE .
                            BUILD missing
                            REMOTE conan-center
                            SETTINGS ${settings})
        list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_BINARY_DIR})
        find_package(OpenVDB REQUIRED)
        target_link_libraries(zsopenvdb INTERFACE ${OpenVDB_LIBRARIES})
        target_link_directories(zsopenvdb INTERFACE ${OpenVDB_LINK_DIRECTORIES})
        target_include_directories(zsopenvdb INTERFACE ${OpenVDB_INCLUDE_DIR})
        # set(ZS_ENABLE_OPENVDB OFF)
    endif()
endif(ZS_ENABLE_OPENVDB)

# cuda
if (ZS_ENABLE_CUDA)
    find_package(CUDAToolkit)
    if (TARGET CUDA::toolkit)
        target_link_libraries(zsproj_deps INTERFACE CUDA::cudart CUDA::cuda_driver CUDA::cublas CUDA::cusparse CUDA::cusolver CUDA::toolkit CUDA::nvrtc)
    else()
        set(ZS_ENABLE_CUDA OFF)
    endif()
endif(ZS_ENABLE_CUDA)

# openmp
if (ZS_ENABLE_OPENMP)
    find_package(OpenMP)
    if (TARGET OpenMP::OpenMP_CXX)
        # https://gitlab.kitware.com/cmake/cmake/-/issues/17256
        # this target already imported in zen
        target_link_libraries(zsproj_deps INTERFACE OpenMP::OpenMP_CXX)
    else()
        set(ZS_ENABLE_OPENMP OFF)
        # message(FATAL_ERROR "OpenMP not found!")
    endif()
endif(ZS_ENABLE_OPENMP)

# threads
if (ZS_ENABLE_PTHREADS)
  find_package(Threads)
  if (NOT TARGET Threads::Threads)
    set(ZS_ENABLE_PTHREADS OFF)
  endif()
  target_link_libraries(zsproj_deps INTERFACE Threads::Threads)
endif(ZS_ENABLE_PTHREADS)

# tpl that needs compilation
# partio
add_subdirectory(partio EXCLUDE_FROM_ALL)
add_dependencies(zsproj_deps zspartio)
target_link_libraries(zsproj_deps INTERFACE zspartio)
#[=[
target_include_directories(zsproj_deps
    INTERFACE   $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/zensim/tpls/partio>
                $<INSTALL_INTERFACE:include/zensim/tpls/partio>
)
#]=]

# fmt
target_compile_definitions(zsproj_deps INTERFACE FMT_HEADER_ONLY)

# spdlog

# Catch2

# gcem

# function_ref

# glm

# jitify

# rapidjson

# cxxopts

# magic_enum

#[=[=
the above are all header-only third-party dependency libraries
#=]=]

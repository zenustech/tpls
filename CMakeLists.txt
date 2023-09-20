cmake_minimum_required(VERSION 3.18)

option(ZS_ENABLE_INSTALL "Enable install rule" OFF)
option(ZS_ENABLE_OPENGL "Enable opengl" OFF)
option(ZS_ENABLE_OPENVDB "Enable openvdb" OFF)
option(ZS_ENABLE_CUDA "Enable cuda" OFF)
option(ZS_ENABLE_VULKAN "Enable vulkan backend" OFF)
option(ZS_ENABLE_SYCL_COMPUTECPP "Enable sycl[computecpp]" OFF)
option(ZS_ENABLE_SYCL_DPCPP "Enable SYCL[Intel-DPC++] backend" OFF)
option(ZS_ENABLE_OPENCL "Enable OpenCL backend" OFF)
option(ZS_ENABLE_OPENMP "Enable openmp" OFF)
option(ZS_ENABLE_JIT "Enable JIT module" OFF)
option(ZS_ENABLE_PTHREADS "Enable pthreads" OFF)
option(ZS_ENABLE_PARTIO "Enable partio" ON)
option(ZS_ENABLE_MSHIO "Enable .msh file io" OFF)

# CUDA
option(ZS_AUTO_DETECT_CUDA_ARCH "Let cmake determine target cuda architecture" ON)
# APPLE
option(USE_MACPORTS "Use MacPorts packages" ON)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMake ${CMAKE_CURRENT_SOURCE_DIR}/CMake ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/CMake ${CMAKE_CURRENT_SOURCE_DIR}/CMake ${CMAKE_CURRENT_BINARY_DIR})

add_library(zpc_deps INTERFACE)
add_library(zsproj_deps INTERFACE)
add_library(zpc_jit_deps INTERFACE)
target_link_libraries(zpc_deps INTERFACE zsproj_deps)

# opengl
if(ZS_ENABLE_OPENGL)
    find_package(OpenGL)

    if(TARGET OpenGL::GL)
        target_link_libraries(zsproj_deps INTERFACE OpenGL::GL)
    else()
        set(ZS_ENABLE_OPENGL OFF)

        # message(FATAL_ERROR "OpenGL not found!")
    endif()
endif(ZS_ENABLE_OPENGL)

set(ZS_ENABLE_OPENGL ${ZS_ENABLE_OPENGL} PARENT_SCOPE)

# cuda
if(ZS_ENABLE_CUDA)
    find_package(CUDAToolkit)

    if(TARGET CUDA::cuda_driver)
        add_library(zpc_cuda_deps INTERFACE)

        if(NOT EXISTS ZS_CUDA_ARCH)
            set(ZS_CUDA_ARCH 52)
        endif()

        # ref: https://gitlab.kitware.com/cmake/cmake/-/issues/19199
        # ref: https://stackoverflow.com/questions/68223398/how-can-i-get-cmake-to-automatically-detect-the-value-for-cuda-architectures
        # ref: https://gitlab.kitware.com/cmake/cmake/-/issues/22375
        if(ZS_AUTO_DETECT_CUDA_ARCH)
            # ref: Kokkos
            # Try to autodetect the CUDA Compute Capability by asking the device
            SET(_BINARY_TEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/CMake/compile_tests/CUDAComputeCapabilityWorkdir)
            FILE(REMOVE_RECURSE ${_BINARY_TEST_DIR})
            FILE(MAKE_DIRECTORY ${_BINARY_TEST_DIR})

            TRY_RUN(
                _RESULT
                _COMPILE_RESULT
                ${_BINARY_TEST_DIR}
                ${CMAKE_CURRENT_SOURCE_DIR}/CMake/compile_tests/cuda_compute_capability.cc
                RUN_OUTPUT_VARIABLE _CUDA_COMPUTE_CAPABILITY)

            IF(NOT _COMPILE_RESULT OR NOT _RESULT EQUAL 0)
                # check to see if CUDA is not already enabled (may happen when Kokkos is subproject)
                GET_PROPERTY(_ENABLED_LANGUAGES GLOBAL PROPERTY ENABLED_LANGUAGES)

                IF(NOT "CUDA" IN_LIST _ENABLED_LANGUAGES)
                    MESSAGE(STATUS "CUDA auto-detection of architecture failed with ${CMAKE_CXX_COMPILER}. Enabling CUDA language ONLY to auto-detect architecture...")
                    INCLUDE(CheckLanguage)
                    CHECK_LANGUAGE(CUDA)

                    IF(CMAKE_CUDA_COMPILER)
                        ENABLE_LANGUAGE(CUDA)
                    ELSE()
                        MESSAGE(STATUS "CUDA language could not be enabled")
                    ENDIF()
                ENDIF()
            ENDIF()

            if(CMAKE_CUDA_COMPILER)
                CONFIGURE_FILE(
                    ${CMAKE_CURRENT_SOURCE_DIR}/CMake/compile_tests/cuda_compute_capability.cc
                    ${PROJECT_BINARY_DIR}/compile_tests/cuda_compute_capability.cu
                    COPYONLY
                )

                set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/CMake/compile_tests/cuda_compute_capability.cc PROPERTIES LANGUAGE CUDA)

                set(CMAKE_CUDA_ARCHITECTURES OFF)
                TRY_RUN(
                    _RESULT
                    _COMPILE_RESULT
                    ${_BINARY_TEST_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/CMake/compile_tests/cuda_compute_capability.cu
                    RUN_OUTPUT_VARIABLE _CUDA_ARCH
                )
                message(STATUS "deduced cuda arch: ${_CUDA_ARCH}")

                set(ZS_CUDA_ARCH ${_CUDA_ARCH})
                # ref: https://zhengqm.github.io/blog/2018/12/07/cuda-nvcc-tips.html
                target_compile_options(zpc_cuda_deps
                    INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_${ZS_CUDA_ARCH},code=compute_${ZS_CUDA_ARCH} -gencode=arch=compute_${ZS_CUDA_ARCH},code=lto_${ZS_CUDA_ARCH}>
                )
                target_link_options(zpc_cuda_deps
                    INTERFACE $<DEVICE_LINK:-arch=compute_${ZS_CUDA_ARCH} -code=sm_${ZS_CUDA_ARCH}> # -dlto -dlink
                )
            endif()
        endif(ZS_AUTO_DETECT_CUDA_ARCH)

        set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} PARENT_SCOPE)
        set(ZS_CUDA_ARCH ${ZS_CUDA_ARCH} PARENT_SCOPE)

        target_link_libraries(zpc_deps INTERFACE zpc_cuda_deps)
        target_compile_features(zpc_cuda_deps INTERFACE cuda_std_17)
        target_link_libraries(zpc_cuda_deps INTERFACE CUDA::cuda_driver CUDA::nvrtc cudadevrt)

        # CUDA::cublas CUDA::cusparse CUDA::cusolver CUDA::cufft
        if(ZS_BUILD_SHARED_LIBS)
            target_link_libraries(zpc_cuda_deps INTERFACE CUDA::cudart)
        else()
            target_link_libraries(zpc_cuda_deps INTERFACE CUDA::cudart_static)
        endif()
    else()
        set(ZS_ENABLE_CUDA OFF)
    endif()
endif(ZS_ENABLE_CUDA)

set(ZS_ENABLE_CUDA ${ZS_ENABLE_CUDA} PARENT_SCOPE)

# vulkan
if(ZS_ENABLE_VULKAN)
  find_package(Vulkan QUIET)

  if(Vulkan_FOUND)
    message("-- vulkan libraries" ${Vulkan_LIBRARIES})
    message("-- vulkan include directory" ${Vulkan_INCLUDE_DIRS})
    message("-- vulkan components" ${Vulkan_FIND_COMPONENTS})
    target_link_libraries(zsproj_deps INTERFACE Vulkan::Vulkan)
    # a not so elegant way of gaining access to other vulkan libs
    get_filename_component(ZS_VULKAN_LINK_DIRECTORY ${Vulkan_LIBRARIES} DIRECTORY)
    message("-- (deduced) vulkan library directory: ${ZS_VULKAN_LINK_DIRECTORY}")
    target_link_directories(zsproj_deps INTERFACE ${ZS_VULKAN_LINK_DIRECTORY})
    target_link_libraries(zsproj_deps INTERFACE shaderc_combined spirv-cross-glsl spirv-cross-core spirv-cross-cpp)

    if(APPLE)
        target_link_libraries(zsproj_deps INTERFACE Vulkan::MoltenVK)
    endif(APPLE)
  else()
    message(STATUS "Vulkan not found!")
    set(ZS_ENABLE_VULKAN OFF)
  endif()
endif(ZS_ENABLE_VULKAN)

set(ZS_ENABLE_VULKAN ${ZS_ENABLE_VULKAN} PARENT_SCOPE)

# JIT
if(ZS_ENABLE_JIT)
    find_package(LLVM 15 CONFIG)
    if (LLVM_FOUND)
        message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
        message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
        message(STATUS "llvm inc dir: ${LLVM_INCLUDE_DIRS}")

        find_package(Clang CONFIG HINTS "${LLVM_DIR}/../clang" "${LLVM_DIR}/../lib/cmake/clang")
        if (Clang_FOUND)
            message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")
            message(STATUS "clang inc dir: ${CLANG_INCLUDE_DIRS}")

            target_compile_definitions(zpc_jit_deps INTERFACE 
                $<BUILD_INTERFACE:${LLVM_DEFINITIONS_LIST}>)
            target_include_directories(zpc_jit_deps INTERFACE 
                $<$<COMPILE_LANGUAGE:CXX>:${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS}>)

            set(CLANG_LIBS 
                clangBasic clangAPINotes clangLex clangParse clangAST clangDynamicASTMatchers clangASTMatchers clangCrossTU clangSema clangCodeGen clangAnalysis clangAnalysisFlowSensitive clangAnalysisFlowSensitiveModels clangEdit clangExtractAPI clangRewrite clangARCMigrate clangDriver clangSerialization clangRewriteFrontend clangFrontend clangFrontendTool clangToolingCore clangToolingInclusions clangToolingRefactoring clangToolingASTDiff clangToolingSyntax clangDependencyScanning clangTransformer clangTooling clangDirectoryWatcher clangIndex clangIndexSerialization clangStaticAnalyzerCore clangStaticAnalyzerCheckers clangStaticAnalyzerFrontend clangFormat clangInterpreter clangHandleCXX clangHandleLLVM)
            
            message(STATUS "llvm libs: ${LLVM_AVAILABLE_LIBS}")
            message(STATUS "clang libs: ${CLANG_LIBS}")

            target_link_libraries(zpc_jit_deps INTERFACE 
                zpc_cxx_deps ${LLVM_AVAILABLE_LIBS} ${CLANG_LIBS})

        else(Clang_FOUND)
            set(ZS_ENABLE_JIT OFF)
        endif(Clang_FOUND)
    else(LLVM_FOUND)
        set(ZS_ENABLE_JIT OFF)
    endif(LLVM_FOUND)
endif(ZS_ENABLE_JIT)

set(ZS_ENABLE_JIT ${ZS_ENABLE_JIT} PARENT_SCOPE)

# sycl-computecpp
if(ZS_ENABLE_SYCL_COMPUTECPP)
    if(NOT DEFINED ComputeCpp_DIR OR NOT DEFINED COMPUTECPP_BITCODE)
        message(WARNING "SYCL enabled but ComputeCpp_DIR and COMPUTECPP_BITCODE not all set!")
        set(ZS_ENABLE_SYCL_COMPUTECPP OFF)
    else(FIND_DIR_MACRO_DEFINED)
        find_package(ComputeCpp)

        if(ComputeCpp_FOUND)
            add_library(zpc_sycl_deps INTERFACE)
            target_link_libraries(zpc_deps INTERFACE zpc_sycl_deps)
            target_compile_features(zpc_sycl_deps INTERFACE cxx_std_17)
            set_target_properties(ComputeCpp::ComputeCpp PROPERTIES IMPORTED_GLOBAL TRUE)
            target_link_libraries(zpc_sycl_deps INTERFACE ComputeCpp::ComputeCpp)

            list(APPEND COMPUTECPP_DEVICE_COMPILER_FLAGS -std=c++17)
            set(COMPUTECPP_DEVICE_COMPILER_FLAGS ${COMPUTECPP_DEVICE_COMPILER_FLAGS} PARENT_SCOPE)
        else(ComputeCpp_NOTFOUND)
            set(ZS_ENABLE_SYCL_COMPUTECPP OFF)
        endif(ComputeCpp_FOUND)
    endif()
endif(ZS_ENABLE_SYCL_COMPUTECPP)

set(ZS_ENABLE_SYCL_COMPUTECPP ${ZS_ENABLE_SYCL_COMPUTECPP} PARENT_SCOPE)

# sycl-dpcpp
if(ZS_ENABLE_SYCL_DPCPP)
    get_filename_component(ZS_CXX_COMPILER_BINARY ${CMAKE_CXX_COMPILER} NAME)

    if(${ZS_CXX_COMPILER_BINARY} STREQUAL "dpcpp")
        add_library(zpc_sycl_deps INTERFACE)
        target_link_libraries(zpc_deps INTERFACE zpc_sycl_deps)
        target_compile_features(zpc_sycl_deps INTERFACE cxx_std_17)
        target_link_libraries(zpc_sycl_deps INTERFACE sycl)
    else()
        set(ZS_ENABLE_SYCL_DPCPP OFF)
    endif()
endif(ZS_ENABLE_SYCL_DPCPP)

set(ZS_ENABLE_SYCL_DPCPP ${ZS_ENABLE_SYCL_DPCPP} PARENT_SCOPE)

# sycl-clang
if(ZS_ENABLE_SYCL)
    if(${CMAKE_CXX_COMPILER_ID} MATCHES ".*Clang")
        add_library(zpc_sycl_deps INTERFACE)
        target_link_libraries(zpc_deps INTERFACE zpc_sycl_deps)
        target_compile_features(zpc_sycl_deps INTERFACE cxx_std_17)
        target_compile_options(zpc_sycl_deps INTERFACE
            $<$<COMPILE_LANGUAGE:CXX>: -fsycl -fsycl-targets=nvptx64>
        )
    else()
        set(ZS_ENABLE_SYCL OFF)
    endif()
endif(ZS_ENABLE_SYCL)

set(ZS_ENABLE_SYCL ${ZS_ENABLE_SYCL} PARENT_SCOPE)

# opencl
if(ZS_ENABLE_OPENCL)
    find_package(OpenCL)

    if(OpenCL_FOUND)
        add_library(zpc_cl_deps INTERFACE)
        target_link_libraries(zpc_deps INTERFACE zpc_cl_deps)

        # target_compile_features(zpc_cl_deps INTERFACE cxx_std_17)
        # target_include_directories(zpc_cl_deps INTERFACE ${OpenCL_INCLUDE_DIRS})
        # target_link_libraries(zpc_cl_deps INTERFACE ${OpenCL_LIBRARIES})
        target_link_libraries(zpc_cl_deps INTERFACE OpenCL::OpenCL)
        message(INFO "opencl directories: ${OpenCL_INCLUDE_DIRS}")
        message(INFO "opencl libraries: ${OpenCL_LIBRARIES}")
    else()
        set(ZS_ENABLE_OPENCL OFF)
    endif()
endif(ZS_ENABLE_OPENCL)

set(ZS_ENABLE_OPENCL ${ZS_ENABLE_OPENCL} PARENT_SCOPE)

# threads
if(ZS_ENABLE_PTHREADS)
    find_package(Threads)

    if(NOT TARGET Threads::Threads)
        set(ZS_ENABLE_PTHREADS OFF)
    endif()

    target_link_libraries(zsproj_deps INTERFACE Threads::Threads)
endif(ZS_ENABLE_PTHREADS)

set(ZS_ENABLE_PTHREADS ${ZS_ENABLE_PTHREADS} PARENT_SCOPE)

# openmp
if(ZS_ENABLE_OPENMP)
    # ref: https://gitlab.com/palisade/palisade-development/-/blob/master/CMakeLists.txt
    if (APPLE)
        if (USE_MACPORTS)
            set(OPENMP_LIBRARIES "/opt/local/lib/libomp")
            set(OPENMP_INCLUDES "/opt/local/include/libomp")
            if(CMAKE_C_COMPILER_ID MATCHES "Clang" OR CMAKE_C_COMPILER_ID MATCHES "AppleClang")
                set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -lomp -Wno-unused-command-line-argument")
                set(OpenMP_C_LIB_NAMES "omp")
                set(OpenMP_omp_LIBRARY ${OpenMP_C_LIB_NAMES})
            endif()
            if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
                set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -lomp -Wno-unused-command-line-argument")
                set(OpenMP_CXX_LIB_NAMES "omp")
                set(OpenMP_omp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
            endif()
        else(USE_MACPORTS)
            if (${ARCHITECTURE} MATCHES "arm64")
                message( STATUS "Apple M1 detected")
                set(OPENMP_LIBRARIES "/opt/homebrew/opt/libomp/lib")
                set(OPENMP_INCLUDES "/opt/homebrew/opt/libomp/include")
            else() # Apple Intel Processor
                message( STATUS "Apple Intel detected")
                set(OPENMP_LIBRARIES "/usr/local/opt/libomp/lib")
                set(OPENMP_INCLUDES "/usr/local/opt/libomp/include")
            endif()

            if(CMAKE_C_COMPILER_ID MATCHES "Clang")
                set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -lomp -Wno-unused-command-line-argument")
                set(OpenMP_C_LIB_NAMES "libomp")
                set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
            endif()
            if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
                set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -lomp -Wno-unused-command-line-argument")
                set(OpenMP_CXX_LIB_NAMES "libomp")
                set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
            endif()
        endif(USE_MACPORTS)
    endif(APPLE)

    find_package(OpenMP)

    if(TARGET OpenMP::OpenMP_CXX)
        add_library(zpc_omp_deps INTERFACE)
        target_link_libraries(zpc_deps INTERFACE zpc_omp_deps)

        # https://gitlab.kitware.com/cmake/cmake/-/issues/17256
        # this target already imported in zen
        target_link_libraries(zpc_omp_deps INTERFACE OpenMP::OpenMP_CXX)
        if (APPLE)
            target_link_directories(zpc_omp_deps INTERFACE ${OPENMP_LIBRARIES})
            target_include_directories(zpc_omp_deps INTERFACE ${OPENMP_INCLUDES})
        endif(APPLE)
    else()
        set(ZS_ENABLE_OPENMP OFF)

        # message(FATAL_ERROR "OpenMP not found!")
    endif()
endif(ZS_ENABLE_OPENMP)

set(ZS_ENABLE_OPENMP ${ZS_ENABLE_OPENMP} PARENT_SCOPE)

# openvdb
if(ZS_ENABLE_OPENVDB)
    if(TARGET openvdb)
        add_library(zsopenvdb INTERFACE)
        target_link_libraries(zpc_deps INTERFACE zsopenvdb)
        target_link_libraries(zsopenvdb INTERFACE openvdb)
    else()
        find_package(OpenVDB)

        if(ZS_BUILD_SHARED_LIBS)
            if(TARGET OpenVDB::openvdb_shared)
                add_library(zsopenvdb INTERFACE)
                target_link_libraries(zpc_deps INTERFACE zsopenvdb)
                target_link_libraries(zsopenvdb INTERFACE OpenVDB::openvdb_shared)
            elseif(TARGET OpenVDB::openvdb)
                add_library(zsopenvdb INTERFACE)
                target_link_libraries(zpc_deps INTERFACE zsopenvdb)
                target_link_libraries(zsopenvdb INTERFACE OpenVDB::openvdb)
            else()
                set(ZS_ENABLE_OPENVDB OFF)
            endif()
        else()
            if(TARGET OpenVDB::openvdb_static)
                add_library(zsopenvdb INTERFACE)
                target_link_libraries(zpc_deps INTERFACE zsopenvdb)
                target_link_libraries(zsopenvdb INTERFACE OpenVDB::openvdb_static)
            elseif(TARGET OpenVDB::openvdb)
                add_library(zsopenvdb INTERFACE)
                target_link_libraries(zpc_deps INTERFACE zsopenvdb)
                target_link_libraries(zsopenvdb INTERFACE OpenVDB::openvdb)
            else()
                set(ZS_ENABLE_OPENVDB OFF)
            endif()
        endif()
    endif()
endif(ZS_ENABLE_OPENVDB)

set(ZS_ENABLE_OPENVDB ${ZS_ENABLE_OPENVDB} PARENT_SCOPE)

# tpl that needs compilation
# whereami
if(ZS_BUILD_SHARED_LIBS)
    set(WHEREAMI_BUILD_SHARED_LIBS ON)
else()
    set(WHEREAMI_BUILD_SHARED_LIBS OFF)
endif()
add_subdirectory(whereami EXCLUDE_FROM_ALL)

# partio
if(ZS_ENABLE_PARTIO)
    if(ZS_BUILD_SHARED_LIBS)
        set(PARTIO_BUILD_SHARED_LIBS ON)
    else()
        set(PARTIO_BUILD_SHARED_LIBS OFF)
    endif()

    add_subdirectory(partio EXCLUDE_FROM_ALL)
endif(ZS_ENABLE_PARTIO)

set(ZS_ENABLE_PARTIO ${ZS_ENABLE_PARTIO} PARENT_SCOPE)

# mshio
if(ZS_ENABLE_MSHIO)
    if(NOT TARGET mshio)
        set(MSHIO_PYTHON OFF)
        set(MSHIO_BUILD_TESTS OFF)
        set(MSHIO_BUILD_EXAMPLES OFF)
        set(MSHIO_EXT_NANOSPLINE OFF)
        add_subdirectory(mshio EXCLUDE_FROM_ALL)
    endif()
endif(ZS_ENABLE_MSHIO)

set(ZS_ENABLE_MSHIO ${ZS_ENABLE_MSHIO} PARENT_SCOPE)

# plog
target_include_directories(zsproj_deps INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

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

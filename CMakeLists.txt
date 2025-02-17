cmake_minimum_required(VERSION 3.18)

option(ZS_ENABLE_INSTALL "Enable install rule" OFF)
option(ZS_ENABLE_OPENGL "Enable opengl" OFF)
option(ZS_ENABLE_OPENVDB "Enable openvdb" OFF)
option(ZS_ENABLE_CUDA "Enable cuda" OFF)
option(ZS_ENABLE_MUSA "Enable musa" OFF)
option(ZS_ENABLE_VULKAN "Enable vulkan backend" OFF)
option(ZS_ENABLE_SYCL_ONEAPI "Enable SYCL[Intel-OneAPI] backend" OFF)
option(ZS_ENABLE_SYCL_ACPP "Enable SYCL[AdaptiveCpp (originally hipSYCL)] backend" OFF)
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
    message(STATUS "\n-- ------------------------------- Checking CUDA ------------------------------- ")
    find_package(CUDAToolkit)

    if(TARGET CUDA::cuda_driver)
        add_library(zpc_cuda_deps INTERFACE)
        add_library(zpc_cuda_arch_options INTERFACE)

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

                set(CMAKE_CUDA_ARCHITECTURES OFF) # this may be questionable
                TRY_RUN(
                    _RESULT
                    _COMPILE_RESULT
                    ${_BINARY_TEST_DIR}
                    ${PROJECT_BINARY_DIR}/compile_tests/cuda_compute_capability.cu
                    LINK_LIBRARIES cuda
                    RUN_OUTPUT_VARIABLE _CUDA_ARCH
                )
                set(ZS_CUDA_ARCH ${_CUDA_ARCH})

                separate_arguments(ZS_CUDA_ARCH)
                message(STATUS "deduced cuda arch: ${ZS_CUDA_ARCH}")

                list(APPEND ZS_CUDA_ARCH 52)

                foreach(CU_ARCH ${ZS_CUDA_ARCH})
                    message(STATUS "appending flag for cuda arch: ${CU_ARCH}")
                    # ref: https://zhengqm.github.io/blog/2018/12/07/cuda-nvcc-tips.html
                    target_compile_options(zpc_cuda_arch_options
                        INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:-gencode=arch=compute_${CU_ARCH},code=compute_${CU_ARCH} -gencode=arch=compute_${CU_ARCH},code=sm_${CU_ARCH}>
                    )
                    target_link_options(zpc_cuda_arch_options
                        INTERFACE $<DEVICE_LINK:-arch=compute_${CU_ARCH} -code=sm_${CU_ARCH}> # -dlto -dlink
                    )
                endforeach()
            endif()
        endif(ZS_AUTO_DETECT_CUDA_ARCH)

        set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES} PARENT_SCOPE)
        set(ZS_CUDA_ARCH ${ZS_CUDA_ARCH} PARENT_SCOPE)

        target_link_libraries(zpc_deps INTERFACE zpc_cuda_deps)
        target_compile_features(zpc_cuda_deps INTERFACE cuda_std_17)
        target_link_libraries(zpc_cuda_deps INTERFACE CUDA::cuda_driver)

        #if (TARGET CUDA::nvJitLink) # introduced in cuda 12
        #    target_link_libraries(zpc_cuda_deps INTERFACE CUDA::nvJitLink)
        #endif()
        # CUDA::cublas CUDA::cusparse CUDA::cusolver CUDA::cufft
        if(ZS_BUILD_SHARED_LIBS)
            target_link_libraries(zpc_cuda_deps INTERFACE CUDA::cudart)
        else()
            target_link_libraries(zpc_cuda_deps INTERFACE CUDA::cudart_static)
        endif()
    else()
        set(ZS_ENABLE_CUDA OFF)
    endif()
    message(STATUS "############################### ZPC: CUDA (${ZS_ENABLE_CUDA}) ###############################")
endif(ZS_ENABLE_CUDA)

set(ZS_ENABLE_CUDA ${ZS_ENABLE_CUDA} PARENT_SCOPE)

# musa
if(ZS_ENABLE_MUSA)
    message(STATUS "\n-- ------------------------------- Checking MUSA ------------------------------- ")

    find_package(MUSA)

    if(MUSA_FOUND)
        add_library(zpc_musa_deps INTERFACE)
        add_library(zpc_musa_arch_options INTERFACE)

        set(_BINARY_TEST_DIR ${CMAKE_CURRENT_BINARY_DIR}/CMake/compile_tests/MUSAComputeCapabilityWorkdir)
        file(REMOVE_RECURSE ${_BINARY_TEST_DIR})
        file(MAKE_DIRECTORY ${_BINARY_TEST_DIR})

        cmake_path(GET MUSA_LIBRARIES PARENT_PATH MUSA_LINK_LIBRARY_DIR)
        # https://stackoverflow.com/questions/19901934/libpthread-so-0-error-adding-symbols-dso-missing-from-command-line 
        execute_process(
            COMMAND ${MUSA_MCC_EXECUTABLE} -x musa ${CMAKE_CURRENT_SOURCE_DIR}/CMake/compile_tests/musa_compute_capability.cc -I${MUSA_INCLUDE_DIRS} -L${MUSA_LINK_LIBRARY_DIR} -Wl,--copy-dt-needed-entries -lmusart -o${_BINARY_TEST_DIR}/test_musa_capability
            RESULT_VARIABLE _RESULT
            OUTPUT_VARIABLE _OUTPUT_RESULT
            ERROR_VARIABLE _ERROR_RESULT
        )
        execute_process(
            COMMAND ${_BINARY_TEST_DIR}/test_musa_capability
            RESULT_VARIABLE _COMPILE_RESULT
            OUTPUT_VARIABLE _MUSA_COMPUTE_CAPABILITY
            ERROR_VARIABLE _ERROR_MESSAGE
        )

        set(ZS_MUSA_ARCH)

        if(_ERROR_RESULT)
            message(STATUS "unable to acquire MUSA compute capability: ${_ERROR_RESULT}")
        else()
            set(ZS_MUSA_ARCH ${_MUSA_COMPUTE_CAPABILITY})
            separate_arguments(ZS_MUSA_ARCH)

            message(STATUS "deduced MUSA compute capability: ${ZS_MUSA_ARCH}")
        endif(_ERROR_RESULT)

        message(STATUS "MUSA_VERSION_MAJOR: ${MUSA_VERSION_MAJOR}")
        message(STATUS "MUSA_VERSION_MINOR: ${MUSA_VERSION_MINOR}")
        message(STATUS "MUSA_VERSION: ${MUSA_VERSION}")
        message(STATUS "MUSA_MCC_EXECUTABLE: ${MUSA_MCC_EXECUTABLE}")
        message(STATUS "MUSA_TOOLKIT_ROOT_DIR: ${MUSA_TOOLKIT_ROOT_DIR}")
        message(STATUS "MUSA_INCLUDE_DIRS: ${MUSA_INCLUDE_DIRS}")
        message(STATUS "MUSA_LIBRARIES: ${MUSA_LIBRARIES}")
        message(STATUS "MUSA_MUFFT_LIBRARIES: ${MUSA_MUFFT_LIBRARIES}")
        message(STATUS "MUSA_MUBLAS_LIBRARIES: ${MUSA_MUBLAS_LIBRARIES}")
        message(STATUS "MUSA_mupti_LIBRARY: ${MUSA_mupti_LIBRARY}")

        foreach(MU_ARCH ${ZS_MUSA_ARCH})
            message(STATUS "appending flag for musa arch: mp_${MU_ARCH}")
            target_compile_options(zpc_musa_arch_options
                INTERFACE --offload-arch=mp_${MU_ARCH}>
            )
        endforeach()

        # https://blog.mthreads.com/blog/musa/2024-05-20-%E4%BD%BF%E7%94%A8cmake%E6%9E%84%E5%BB%BAMUSA%E5%B7%A5%E7%A8%8B/
        target_link_libraries(zpc_deps INTERFACE zpc_musa_deps)
        # target_include_directories(zpc_musa_deps INTERFACE ${MUSA_INCLUDE_DIRS})
        target_link_libraries(zpc_musa_deps INTERFACE MUSA::musart)
        target_compile_options(zpc_musa_deps INTERFACE -fPIC -mtgpu-enable-const-calc)
    else()
        set(ZS_ENABLE_MUSA OFF)
    endif(MUSA_FOUND)
    message(STATUS "############################### ZPC: MUSA (${ZS_ENABLE_MUSA}) ###############################")
endif(ZS_ENABLE_MUSA)

set(ZS_ENABLE_MUSA ${ZS_ENABLE_MUSA} PARENT_SCOPE)

# vulkan
if(ZS_ENABLE_VULKAN)
    message(STATUS "\n-- ------------------------------- Checking Vulkan ------------------------------- ")
    find_package(Vulkan QUIET)

    if(Vulkan_FOUND)
        message("-- vulkan version: ${Vulkan_VERSION}")
        message("-- vulkan libraries: " ${Vulkan_LIBRARIES})
        message("-- vulkan include directories: " ${Vulkan_INCLUDE_DIRS})
        message("-- vulkan include directory: " ${Vulkan_INCLUDE_DIR})
        message("-- vulkan components: " ${Vulkan_FIND_COMPONENTS})
        target_link_libraries(zsproj_deps INTERFACE Vulkan::Vulkan)
        # a not so elegant way of gaining access to other vulkan libs
        get_filename_component(ZS_VULKAN_LINK_DIRECTORY ${Vulkan_LIBRARIES} DIRECTORY)
        message("-- (deduced) vulkan library directory: ${ZS_VULKAN_LINK_DIRECTORY}")
        target_link_directories(zsproj_deps INTERFACE ${ZS_VULKAN_LINK_DIRECTORY})

        set(VK_ADDITIONAL_LIBS shaderc_combined spirv-cross-glsl spirv-cross-core spirv-cross-cpp)
        set(AVAILABLE_VK_ADDITIONAL_LIBS "")
        foreach(lib ${VK_ADDITIONAL_LIBS})
            set(found False)
            foreach(suffix ${CMAKE_FIND_LIBRARY_SUFFIXES})
                set(libfile "")
                if(CMAKE_BUILD_TYPE STREQUAL "Debug")
                    set(libfile ${ZS_VULKAN_LINK_DIRECTORY}/${CMAKE_SHARED_LIBRARY_PREFIX}${lib}d${suffix})
                else()
                    set(libfile ${ZS_VULKAN_LINK_DIRECTORY}/${CMAKE_SHARED_LIBRARY_PREFIX}${lib}${suffix})
                endif()
                if(EXISTS ${libfile})
                    if(found)
                        message(FATAL_ERROR "Multiple library types of [${lib}] is found! Unable to determine.")
                    endif(found)
                    list(APPEND AVAILABLE_VK_ADDITIONAL_LIBS ${libfile})
                    set(found True)
                elseif(CMAKE_BUILD_TYPE STREQUAL "Debug") # try non-debug version
                    set(libfile ${ZS_VULKAN_LINK_DIRECTORY}/${CMAKE_SHARED_LIBRARY_PREFIX}${lib}${suffix})
                    if(EXISTS ${libfile})
                        list(APPEND AVAILABLE_VK_ADDITIONAL_LIBS ${libfile})
                        set(found True)
                    endif()
                endif()
            endforeach(suffix ${CMAKE_FIND_LIBRARY_SUFFIXES})
        endforeach(lib ${VK_ADDITIONAL_LIBS})
        message("-- additional available libraries: ${AVAILABLE_VK_ADDITIONAL_LIBS}")
        target_link_libraries(zsproj_deps INTERFACE ${AVAILABLE_VK_ADDITIONAL_LIBS})
        #[=[
    target_link_libraries(zsproj_deps INTERFACE 
      $<IF:$<CONFIG:Debug>, 
        shaderc_combinedd spirv-cross-glsld spirv-cross-cored spirv-cross-cppd, 
        shaderc_combined spirv-cross-glsl spirv-cross-core spirv-cross-cpp>)
    ]=]
        # target_include_directories(zsproj_deps INTERFACE "$ENV{VULKAN_SDK}/include")
        target_include_directories(zsproj_deps INTERFACE ${Vulkan_INCLUDE_DIRS})
        if(${Vulkan_VERSION} GREATER_EQUAL 1.4.0.0)
            target_compile_definitions(zsproj_deps INTERFACE
                -DZS_VK_DYNAMIC_LOADER=vk::detail::DynamicLoader
                -DZS_VK_DISPATCH_LOADER_DYNAMIC=vk::detail::DispatchLoaderDynamic
            )
        else()
            target_compile_definitions(zsproj_deps INTERFACE
                -DZS_VK_DYNAMIC_LOADER=vk::DynamicLoader
                -DZS_VK_DISPATCH_LOADER_DYNAMIC=vk::DispatchLoaderDynamic
            )
        endif()

        if(NOT EXISTS ${Vulkan_INCLUDE_DIR}/vma/vk_mem_alloc.h)
            target_include_directories(zsproj_deps INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/vma")
        endif()

        # if(APPLE)
        #     target_link_libraries(zsproj_deps INTERFACE Vulkan::MoltenVK)
        # endif(APPLE)
    else()
        message(STATUS "Vulkan not found!")
        set(ZS_ENABLE_VULKAN OFF)
    endif()
    message(STATUS "############################### ZPC: Vulkan (${ZS_ENABLE_VULKAN}) ###############################")
endif(ZS_ENABLE_VULKAN)

set(ZS_ENABLE_VULKAN ${ZS_ENABLE_VULKAN} PARENT_SCOPE)

# JIT
if(ZS_ENABLE_JIT)
    message(STATUS "\n-- ------------------------------- Checking LLVM/Clang ------------------------------- ")
    find_package(LLVM CONFIG)
    if(LLVM_FOUND)
        message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
        message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
        message(STATUS "llvm inc dir: ${LLVM_INCLUDE_DIRS}")

        find_package(Clang CONFIG HINTS "${LLVM_DIR}/../clang" "${LLVM_DIR}/../lib/cmake/clang")
        if(Clang_FOUND AND ${LLVM_PACKAGE_VERSION} GREATER_EQUAL 15)
            message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")
            message(STATUS "clang inc dir: ${CLANG_INCLUDE_DIRS}")

            separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
            target_compile_definitions(zpc_jit_deps INTERFACE
                $<BUILD_INTERFACE: ${LLVM_DEFINITIONS_LIST} LLVM_VERSION_MAJOR=${LLVM_VERSION_MAJOR}>)
            target_include_directories(zpc_jit_deps INTERFACE
                $<$<COMPILE_LANGUAGE:CXX>: ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS}>)

            llvm_map_components_to_libnames(llvm_libs support core irreader)

            set(CLANG_LIBS
                libclang clangBasic clangAPINotes clangLex clangParse clangAST clangDynamicASTMatchers clangASTMatchers clangCrossTU clangSema clangCodeGen clangAnalysis clangAnalysisFlowSensitive clangAnalysisFlowSensitiveModels clangEdit clangExtractAPI clangRewrite clangDriver clangSerialization clangRewriteFrontend clangFrontend clangFrontendTool clangToolingCore clangToolingInclusions clangToolingRefactoring clangToolingASTDiff clangToolingSyntax clangDependencyScanning clangTransformer clangTooling clangDirectoryWatcher clangIndex clangIndexSerialization clangStaticAnalyzerCore clangStaticAnalyzerCheckers clangStaticAnalyzerFrontend clangFormat clangInterpreter clangHandleCXX clangHandleLLVM)
            # libclang is the only lib without prefix
            set(CLANG_AVAILABLE_LIBS "libclang")
            set(CLANG_AVAILABLE_LIB_FILES "libclang")
            foreach(lib ${CLANG_LIBS})
                if(NOT lib STREQUAL "libclang")
                    set(found False)
                    foreach(suffix ${CMAKE_FIND_LIBRARY_SUFFIXES})
                        set(libfile ${LLVM_LIBRARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}${lib}${suffix})
                        if(EXISTS ${libfile})
                            if(found)
                                message(FATAL_ERROR "Multiple library types of [${lib}] is found! Unable to determine.")
                            endif(found)
                            list(APPEND CLANG_AVAILABLE_LIBS ${lib})
                            list(APPEND CLANG_AVAILABLE_LIB_FILES ${libfile})
                            set(found True)
                        endif(EXISTS ${libfile})
                    endforeach(suffix ${CMAKE_FIND_LIBRARY_SUFFIXES})
                endif(NOT lib STREQUAL "libclang")
            endforeach(lib ${CLANG_LIBS})

            message(STATUS "llvm compile definitions: ${LLVM_DEFINITIONS_LIST}")
            message(STATUS "available llvm libs: ${LLVM_AVAILABLE_LIBS}")
            message(STATUS "required llvm libs: ${llvm_libs}")
            message(STATUS "available clang libs: ${CLANG_AVAILABLE_LIBS}")

            target_link_directories(zpc_jit_deps INTERFACE ${LLVM_LIBRARY_DIR})
            target_link_libraries(zpc_jit_deps INTERFACE
                zpc_cxx_deps ${llvm_libs} ${CLANG_AVAILABLE_LIBS})

        else(Clang_FOUND AND ${LLVM_PACKAGE_VERSION} GREATER_EQUAL 15)
            message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} at ${LLVM_DIR}, Clang package search status: ${Clang_FOUND}.")
            set(ZS_ENABLE_JIT OFF)
        endif(Clang_FOUND AND ${LLVM_PACKAGE_VERSION} GREATER_EQUAL 15)
    else(LLVM_FOUND)
        set(ZS_ENABLE_JIT OFF)
    endif(LLVM_FOUND)
    message(STATUS "############################### ZPC: LLVM/Clang (${ZS_ENABLE_JIT}) ###############################")
endif(ZS_ENABLE_JIT)

set(ZS_ENABLE_JIT ${ZS_ENABLE_JIT} PARENT_SCOPE)
if(ZS_ENABLE_JIT)
    set(LLVM_VERSION_MAJOR ${LLVM_VERSION_MAJOR} PARENT_SCOPE)
endif()

# intel-sycl
if(ZS_ENABLE_SYCL_ONEAPI)
    message(STATUS "\n-- ------------------------------- Checking Intel SYCL ------------------------------- ")
    # get_filename_component(ZS_CXX_COMPILER_BINARY ${CMAKE_CXX_COMPILER} NAME)
    cmake_path(GET CMAKE_CXX_COMPILER STEM LAST_ONLY ZS_CXX_COMPILER_BINARY)
    message(STATUS "Current C++ compiler is ${ZS_CXX_COMPILER_BINARY} (${CMAKE_CXX_COMPILER})")

    find_package(IntelSYCL)

    if(IntelSYCL_FOUND)
        if(${ZS_CXX_COMPILER_BINARY} STREQUAL "icx")
            add_library(zpc_sycl_deps INTERFACE)
            target_link_libraries(zpc_deps INTERFACE zpc_sycl_deps)
            target_compile_features(zpc_sycl_deps INTERFACE cxx_std_17)
            target_link_libraries(zpc_sycl_deps INTERFACE IntelSYCL::SYCL_CXX)
            # target_link_directories(zpc_sycl_deps INTERFACE ${SYCL_LIBRARY_DIR})
            ## ref: https://www.cntofu.com/book/46/linux_system/1515.md
            if(NOT WIN32)
                target_link_options(zpc_sycl_deps INTERFACE $<HOST_LINK:-rpath,${SYCL_LIBRARY_DIR}>)
            else()
                target_link_options(zpc_sycl_deps INTERFACE $<HOST_LINK: /LIBPATH:${SYCL_LIBRARY_DIR}>)
            endif()
            # message(STATUS "\n\n lib dir: ${SYCL_LIBRARY_DIR} \n\n")
            # message(STATUS "\n\n link flags: ${SYCL_LINK_FLAGS} \n\n")
            # message(STATUS "\n\n compile flags: ${SYCL_FLAGS} \n\n")
            # message(STATUS "\n\n inc dir: ${SYCL_INCLUDE_DIR} \n\n")
        else()
            message(STATUS "To use IntelSYCL, please add \"-DCMAKE_C_COMPILER=\"path/to/compiler/icx\" -DCMAKE_CXX_COMPILER=\"path/to/compiler/icx\"\"")
            set(ZS_ENABLE_SYCL_ONEAPI OFF)
        endif()
    else(IntelSYCL_FOUND)
        set(ZS_ENABLE_SYCL_ONEAPI OFF)
        message(STATUS "To enable IntelSYCL, please add \"-DIntelSYCL_DIR=\"path/where/IntelSYCLConfig.cmake/reisdes\" -DCMAKE_C_COMPILER=\"path/to/compiler/icx\" -DCMAKE_CXX_COMPILER=\"path/to/compiler/icx\"\" upon cmake configuration")
        if(WIN32)
            message(STATUS "On windows, please open \"Intel oneAPI command prompt for Intel 64 for Visual Studio 2022\" and type \"devenv\" to open Visual Studio IDE!")
        endif(WIN32)
    endif(IntelSYCL_FOUND)
    message(STATUS "############################### ZPC: Intel SYCL (${ZS_ENABLE_SYCL_ONEAPI}) ###############################")
endif(ZS_ENABLE_SYCL_ONEAPI)

set(ZS_ENABLE_SYCL_ONEAPI ${ZS_ENABLE_SYCL_ONEAPI} PARENT_SCOPE)

# sycl-clang
if(ZS_ENABLE_SYCL_ACPP)
    message(STATUS "\n-- ------------------------------- Checking AdaptiveCpp SYCL ------------------------------- ")
    cmake_path(GET CMAKE_CXX_COMPILER STEM LAST_ONLY ZS_CXX_COMPILER_BINARY)
    message(STATUS "Current C++ compiler is ${ZS_CXX_COMPILER_BINARY} (${CMAKE_CXX_COMPILER})")

    find_package(AdaptiveCpp CONFIG)

    # if(${CMAKE_CXX_COMPILER_ID} MATCHES ".*Clang")
    if(AdaptiveCpp_FOUND)
        add_library(zpc_sycl_deps INTERFACE)
        target_link_libraries(zpc_deps INTERFACE zpc_sycl_deps)
        target_compile_features(zpc_sycl_deps INTERFACE cxx_std_17)
        target_link_libraries(zpc_sycl_deps INTERFACE AdaptiveCpp::acpp-rt)
    else(AdaptiveCpp_FOUND)
        set(ZS_ENABLE_SYCL_ACPP OFF)
        message(STATUS "To enable AdaptiveCpp, please add \"-DAdaptiveCpp_DIR=\"path/where/adaptivecpp-config.cmake/reisdes\" upon cmake configuration")
    endif(AdaptiveCpp_FOUND)
    message(STATUS "############################### ZPC: AdaptiveCpp SYCL (${ZS_ENABLE_SYCL_ACPP}) ###############################")
endif(ZS_ENABLE_SYCL_ACPP)

set(ZS_ENABLE_SYCL_ACPP ${ZS_ENABLE_SYCL_ACPP} PARENT_SCOPE)

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
    message(STATUS "\n-- ------------------------------- Checking OpenMP ------------------------------- ")
    # ref: https://gitlab.com/palisade/palisade-development/-/blob/master/CMakeLists.txt
    if(APPLE)
        if(USE_MACPORTS)
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
            execute_process(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE)
            if(${ARCHITECTURE} MATCHES "arm64")
                message(STATUS "Apple M1 detected")
                set(OPENMP_LIBRARIES "/opt/homebrew/opt/libomp/lib")
                set(OPENMP_INCLUDES "/opt/homebrew/opt/libomp/include")
            else() # Apple Intel Processor
                message(STATUS "Apple Intel detected")
                set(OPENMP_LIBRARIES "/usr/local/opt/libomp/lib")
                set(OPENMP_INCLUDES "/usr/local/opt/libomp/include")
            endif()

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
        endif(USE_MACPORTS)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
        if(WIN32)
            ## ref: https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2023-0/use-the-openmp-libraries.html
            set(OpenMP_C_FLAGS "/Qopenmp")
            set(OpenMP_CXX_FLAGS "/Qopenmp")
            set(OpenMP_C_LIB_NAMES "libiomp5md")
            set(OpenMP_libiomp5md_LIBRARY ${OpenMP_C_LIB_NAMES})
            set(OpenMP_CXX_LIB_NAMES "libiomp5md")
            set(OpenMP_libiomp5md_LIBRARY ${OpenMP_CXX_LIB_NAMES})
        else()
            set(OpenMP_C_FLAGS "-qopenmp")
            set(OpenMP_CXX_FLAGS "-qopenmp")
            set(OpenMP_C_LIB_NAMES "iomp5")
            set(OpenMP_iomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
            set(OpenMP_CXX_LIB_NAMES "iomp5")
            set(OpenMP_iomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
        endif(WIN32)
    endif(APPLE)

    if(OPENMP_INCLUDES)
        set(OpenMP_CXX_INCLUDE_DIR ${OPENMP_INCLUDES})
    endif()
    find_package(OpenMP)

    if(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
        add_library(zpc_omp_deps INTERFACE)
        target_link_libraries(zpc_deps INTERFACE zpc_omp_deps)
        #[=[
        target_compile_options(zpc_omp_deps 
            INTERFACE $<$<COMPILE_LANGUAGE:CXX>: $<IF:$<BOOL:${WIN32}>, 
                /Qiopenmp /Qopenmp-targets:spir64, 
                -fiopenmp -fopenmp-targets=spir64>>
        )]=]
        target_link_libraries(zpc_omp_deps INTERFACE OpenMP::OpenMP_CXX)
        # https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2023-0/fopenmp-targets-qopenmp-targets.html
        # /Qopenmp-targets:spir64 -fopenmp-targets=spir64
    elseif(TARGET OpenMP::OpenMP_CXX)
        add_library(zpc_omp_deps INTERFACE)
        target_link_libraries(zpc_deps INTERFACE zpc_omp_deps)

        # https://gitlab.kitware.com/cmake/cmake/-/issues/17256
        # this target already imported in zen
        target_link_libraries(zpc_omp_deps INTERFACE OpenMP::OpenMP_CXX)
        if(APPLE)
            target_link_directories(zpc_omp_deps INTERFACE ${OPENMP_LIBRARIES})
            target_include_directories(zpc_omp_deps INTERFACE ${OPENMP_INCLUDES})
        endif(APPLE)
    else()
        set(ZS_ENABLE_OPENMP OFF)

        # message(FATAL_ERROR "OpenMP not found!")
    endif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")

    message(STATUS "############################### ZPC: OpenMP (${ZS_ENABLE_OPENMP}) ###############################")
endif(ZS_ENABLE_OPENMP)

set(ZS_ENABLE_OPENMP ${ZS_ENABLE_OPENMP} PARENT_SCOPE)

# openvdb
if(ZS_ENABLE_OPENVDB)
    if(TARGET openvdb)
        add_library(zsopenvdb INTERFACE)
        target_link_libraries(zpc_deps INTERFACE zsopenvdb)
        target_link_libraries(zsopenvdb INTERFACE openvdb)
    else()
        list(APPEND CMAKE_MODULE_PATH /usr/local/lib/cmake/OpenVDB/) # openvdb's default linux install location
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
if(NOT DEFINED WHEREAMI_BUILD_SHARED_LIBS)
    if(ZS_BUILD_SHARED_LIBS)
        set(WHEREAMI_BUILD_SHARED_LIBS ON)
    else()
        set(WHEREAMI_BUILD_SHARED_LIBS OFF)
    endif(ZS_BUILD_SHARED_LIBS)
endif(NOT DEFINED WHEREAMI_BUILD_SHARED_LIBS)
add_subdirectory(whereami EXCLUDE_FROM_ALL)

# partio
if(ZS_ENABLE_PARTIO)
    if(NOT DEFINED PARTIO_BUILD_SHARED_LIBS)
        if(ZS_BUILD_SHARED_LIBS)
            set(PARTIO_BUILD_SHARED_LIBS ON)
        else()
            set(PARTIO_BUILD_SHARED_LIBS OFF)
        endif(ZS_BUILD_SHARED_LIBS)
    endif(NOT DEFINED PARTIO_BUILD_SHARED_LIBS)

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

# asio
target_include_directories(zsproj_deps INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/asio
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>/asio
)

# other local tpls
# fmt
target_compile_definitions(zsproj_deps INTERFACE -DFMT_HEADER_ONLY=1)

# Catch2
# gcem
# function_ref
# jitify
# rapidjson
# cxxopts
# magic_enum

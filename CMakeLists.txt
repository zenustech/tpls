cmake_minimum_required(VERSION 3.18)

project(ZSDEPS CXX)

option(ZS_ENABLE_INSTALL "Enable install rule" OFF)
option(ZS_ENABLE_OPENGL "Enable opengl" ON)
option(ZS_ENABLE_OPENVDB "Enable openvdb" ON)
option(ZS_ENABLE_CUDA "Enable cuda" ON)
option(ZS_ENABLE_OPENMP "Enable openmp" ON)
option(ZS_ENABLE_PTHREADS "Enable pthreads" ON)
option(ZS_ENABLE_PARTIO "Enable partio" ON)

list(APPEND CMAKE_MODULE_PATH
  ${PROJECT_SOURCE_DIR}/CMake
)

add_library(zsproj_deps INTERFACE)
target_include_directories(zsproj_deps INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/>
)

# opengl
if (ZS_ENABLE_OPENGL)
    find_package(OpenGL)
    if (TARGET OpenGL::GL)
        get_target_property(OPENGL_IMPORT_STATUS OpenGL::GL IMPORTED_GLOBAL)
        if (NOT OPENGL_IMPORT_STATUS)
            set_target_properties(OpenGL::GL PROPERTIES IMPORTED_GLOBAL TRUE)
        endif()
        target_link_libraries(zsproj_deps INTERFACE OpenGL::GL)
    else()
        set(ZS_ENABLE_OPENGL OFF)
        # message(FATAL_ERROR "OpenGL not found!")
    endif()
endif(ZS_ENABLE_OPENGL)

# openvdb
if (ZS_ENABLE_OPENVDB)
    list(APPEND CMAKE_MODULE_PATH /usr/local/lib/cmake/OpenVDB/)
    find_package(OpenVDB)
    if (TARGET OpenVDB::openvdb)
        get_target_property(OPENVDB_IMPORT_STATUS OpenVDB::openvdb IMPORTED_GLOBAL)
        if (NOT OPENVDB_IMPORT_STATUS)
            set_target_properties(OpenVDB::openvdb PROPERTIES IMPORTED_GLOBAL TRUE)
        endif()
        # target_link_libraries(zsproj_deps INTERFACE OpenVDB::openvdb)
    else()
        set(ZS_ENABLE_OPENVDB OFF)
    endif()
endif(ZS_ENABLE_OPENVDB)

# cuda
if (ZS_ENABLE_CUDA)
    find_package(CUDAToolkit)
    if (TARGET CUDA::cudart)
        target_link_libraries(zsproj_deps INTERFACE CUDA::cudart CUDA::cuda_driver CUDA::cublas CUDA::cusparse CUDA::cusolver)
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
        get_target_property(OPENMP_IMPORT_STATUS OpenMP::OpenMP_CXX IMPORTED_GLOBAL)
        if (NOT OPENMP_IMPORT_STATUS)
            set_target_properties(OpenMP::OpenMP_CXX PROPERTIES IMPORTED_GLOBAL TRUE)
        endif()
        target_link_libraries(zsproj_deps INTERFACE OpenMP::OpenMP_CXX)
    else()
        set(ZS_ENABLE_OPENMP OFF)
        # message(FATAL_ERROR "OpenMP not found!")
    endif()
endif(ZS_ENABLE_OPENMP)

# threads
if(UNIX)
if (ZS_ENABLE_PTHREADS)
  find_package(Threads)
  if (NOT TARGET Threads::Threads)
    set(ZS_ENABLE_PTHREADS OFF)
  endif()
  get_target_property(THREADS_IMPORT_STATUS Threads::Threads IMPORTED_GLOBAL)
  if (NOT THREADS_IMPORT_STATUS)
    set_target_properties(Threads::Threads PROPERTIES IMPORTED_GLOBAL TRUE)
  endif()
  target_link_libraries(zsproj_deps INTERFACE Threads::Threads)
endif(ZS_ENABLE_PTHREADS)
endif()

# tpl that needs compilation
# partio
add_subdirectory(partio EXCLUDE_FROM_ALL)
add_dependencies(zsproj_deps zspartio)
target_link_libraries(zsproj_deps INTERFACE zspartio)
target_include_directories(zsproj_deps
    INTERFACE   $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/zensim/tpls/partio>
                $<INSTALL_INTERFACE:include/zensim/tpls/partio>
)

# umpire
if (ZS_ENABLE_CUDA)
    set(ENABLE_CUDA On CACHE BOOL "Enable CUDA for Umpire build")
endif()
if (ZS_ENABLE_OPENMP)
    set(ENABLE_OPENMP On CACHE BOOL "Enable CUDA for Umpire build")
endif()
add_subdirectory(umpire EXCLUDE_FROM_ALL)
add_dependencies(zsproj_deps zsumpire)
target_link_libraries(zsproj_deps INTERFACE zsumpire)
if (ZS_ENABLE_CUDA)
    unset(ENABLE_CUDA)
endif()
if (ZS_ENABLE_OPENMP)
    unset(ENABLE_OPENMP)
endif()

# fmt
target_compile_definitions(zsproj_deps INTERFACE FMT_HEADER_ONLY)
target_include_directories(zsproj_deps INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/fmt/>
    $<INSTALL_INTERFACE:include/zensim/tpls/fmt/>
)

# spdlog
target_include_directories(zsproj_deps INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/spdlog/>
    $<INSTALL_INTERFACE:include/zensim/tpls/spdlog/>
)

# Catch2
target_include_directories(zsproj_deps INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/catch2/>
    $<INSTALL_INTERFACE:include/zensim/tpls/catch2/>
)

# gcem
target_include_directories(zsproj_deps INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/gcem/>
    $<INSTALL_INTERFACE:include/zensim/tpls/gcem/>
)

# function_ref
target_include_directories(zsproj_deps INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/function_ref/>
    $<INSTALL_INTERFACE:include/zensim/tpls/function_ref/>
)

# glm
target_include_directories(zsproj_deps INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/glm/>
    $<INSTALL_INTERFACE:include/zensim/tpls/glm/>
)

# jitify
target_include_directories(zsproj_deps INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/jitify/>
    $<INSTALL_INTERFACE:include/zensim/tpls/jitify/>
)

# rapidjson
target_include_directories(zsproj_deps INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/rapidjson>
    $<INSTALL_INTERFACE:include/zensim/tpls/rapidjson>
)

# cxxopts
target_include_directories(zsproj_deps INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cxxopts>
    $<INSTALL_INTERFACE:include/zensim/tpls/cxxopts>
)

# magic_enum
target_include_directories(zsproj_deps INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/magic_enum>
    $<INSTALL_INTERFACE:include/zensim/tpls/magic_enum>
)

#[=[=
the above are all header-only third-party dependency libraries
#=]=]

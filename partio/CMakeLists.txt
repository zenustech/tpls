# PARTIO SOFTWARE
# Copyright 2010 Disney Enterprises, Inc. All rights reserved
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
# 
# * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# 
# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in
# the documentation and/or other materials provided with the
# distribution.
# 
# * The names "Disney", "Walt Disney Pictures", "Walt Disney Animation
# Studios" or the names of its contributors may NOT be used to
# endorse or promote products derived from this software without
# specific prior written permission from Walt Disney Pictures.
# 
# Disclaimer: THIS SOFTWARE IS PROVIDED BY WALT DISNEY PICTURES AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
# BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE, NONINFRINGEMENT AND TITLE ARE DISCLAIMED.
# IN NO EVENT SHALL WALT DISNEY PICTURES, THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND BASED ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

cmake_minimum_required(VERSION 3.15.0)
project(partio LANGUAGES CXX)
set(PARTIO_VERSION_MAJOR "1")

if (DEFINED ENV{CXXFLAGS_STD})
    set(CXXFLAGS_STD $ENV{CXXFLAGS_STD})
else()
    set(CXXFLAGS_STD "c++14")
endif()
# Transform "c++17" into "17".  "-1" means "rest of the string".
if (NOT DEFINED WDAS_CXX_STANDARD)
    string(SUBSTRING "${CXXFLAGS_STD}" 3 -1 WDAS_CXX_STANDARD)
endif()

set(CMAKE_CXX_STANDARD "${WDAS_CXX_STANDARD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_FIND_PACKAGE_RESOLVE_SYMLINKS ON)
set(CMAKE_THREAD_PREFER_PTHREAD ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)

if (WIN32)
    if (MSVC)
        add_definitions(-DPARTIO_WIN32)
    endif()
    add_definitions(-D_USE_MATH_DEFINES)
else()
    add_compile_options(-Wextra -Wno-unused-parameter)
endif()

if (APPLE)
  set(CMAKE_SHARED_LINKER_FLAGS "-undefined dynamic_lookup")
  add_definitions(-DGL_SILENCE_DEPRECATION)
endif()

find_package(ZLIB)
if (ZLIB_FOUND)
    add_definitions(-DPARTIO_USE_ZLIB)
else()
    set(ZLIB_LIBRARY "")
endif()

# Make modules able to see partio library
set(PARTIO_LIBRARIES zspartio ${ZLIB_LIBRARY})
 

# build zspartio
file(GLOB io_cpp "io/*.cpp")
file(GLOB core_cpp "core/*.cpp")

add_library(zspartio STATIC ${PARTIO_LIBRARY_TYPE} ${io_cpp} ${core_cpp})
set_target_properties(zspartio PROPERTIES
    OUTPUT_NAME zspartio
    POSITION_INDEPENDENT_CODE ON
    SOVERSION ${PARTIO_VERSION_MAJOR})

target_include_directories(zspartio
    PUBLIC
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

if (ZLIB_FOUND)
    target_link_libraries(zspartio PUBLIC ZLIB::ZLIB)
endif()

file(GLOB public_includes "*.h")

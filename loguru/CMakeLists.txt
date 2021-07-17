cmake_minimum_required(VERSION 3.18)

project(loguru)

MESSAGE(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

add_library(zsloguru loguru.cpp)

target_compile_options(zsloguru
  PRIVATE       $<$<COMPILE_LANGUAGE:CXX>: $<IF:$<CXX_COMPILER_ID:MSVC>, 
  		/Wall,
  		-std=c++11 -Werror -Wall -Wextra -pedantic 
		$<IF:$<CXX_COMPILER_ID:Clang>, -Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic 
			-Wno-missing-noreturn -Wno-missing-prototypes 
			-Wno-gnu-zero-variadic-macro-arguments -Wno-disabled-macro-expansion, > >>
)

find_package(Threads)
target_link_libraries(zsloguru Threads::Threads) # For pthreads
if(NOT WIN32)
    target_link_libraries(zsloguru dl)
endif()

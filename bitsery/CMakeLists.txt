cmake_minimum_required(VERSION 3.1)
project(bitsery
        LANGUAGES CXX
        VERSION 5.2.2)

#============= setup target ======================
add_library(bitsery INTERFACE)
# create alias, so that user could always write target_link_libraries(... Bitsery::bitsery)
# despite of bitsery target is imported or not
add_library(Bitsery::bitsery ALIAS bitsery)

target_include_directories(bitsery INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/../)
target_compile_features(bitsery INTERFACE
        cxx_auto_type
        cxx_constexpr
        cxx_lambdas
        cxx_nullptr
        cxx_variadic_templates)


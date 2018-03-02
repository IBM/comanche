cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include(./.cmakeinclude)

project(blob-tests CXX)

find_library(GTEST_LIB NAMES GTEST PATHS /usr/local/lib,/usr/lib)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

add_executable(blob-test1 test1.cpp)
target_link_libraries(blob-test1 gtest pthread common dl comanche-blob numa)
target_compile_features(blob-test1 PRIVATE cxx_range_for)

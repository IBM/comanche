cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include(./.cmakeinclude)

project(rdma-tests CXX)
 
find_library(GTEST_LIB NAMES GTEST PATHS /usr/local/lib,/usr/lib)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

add_executable(fabric-test1 test1.cpp)
target_link_libraries(fabric-test1 ${ASAN_LIB} common comanche-core fabric gtest pthread dl) # dl

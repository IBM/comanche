cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


project(log-store-tests CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

find_library(GTEST_LIB NAMES GTEST PATHS /usr/local/lib,/usr/lib)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

add_executable(log-store-test1 test1.cpp)
target_link_libraries(log-store-test1 ${ASAN_LIB} common comanche-core numa gtest pthread dl comanche-storelog)
target_compile_features(log-store-test1 PRIVATE cxx_range_for)

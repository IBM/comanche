cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(pmem-posix-tests CXX)


find_library(GTEST_LIB NAMES GTEST PATHS /usr/local/lib,/usr/lib)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

add_executable(pmem-posix-test1 test1.cpp)
target_link_libraries(pmem-posix-test1 common numa gtest pthread dl comanche-pmempaged)
target_compile_features(pmem-posix-test1 PRIVATE cxx_range_for)

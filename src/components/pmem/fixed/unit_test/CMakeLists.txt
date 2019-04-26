cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


project(pmem-fixed-tests CXX)

find_library(GTEST_LIB NAMES GTEST PATHS /usr/local/lib,/usr/lib)

link_directories(/usr/local/lib64)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

add_executable(pmem-fixed-test1 test1.cpp)
target_link_libraries(pmem-fixed-test1 ${ASAN_LIB} common numa gtest pthread dl comanche-pmemfixed)
target_compile_features(pmem-fixed-test1 PRIVATE cxx_range_for)

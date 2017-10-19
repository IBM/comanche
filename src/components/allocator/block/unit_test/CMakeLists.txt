cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include($ENV{COMANCHE_HOME}/mk/common.cmake)

project(block-allocator-tests CXX)
 
find_library(GTEST_LIB NAMES GTEST PATHS /usr/local/lib,/usr/lib)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

# Including extra cmake rules
#include(../../../../mk/clang-dev-tools.cmake)

add_executable(block-allocator-test1 test1.cpp)
target_link_libraries(block-allocator-test1 ${ASAN_LIB} common numa gtest pthread dl comanche-allocblock)
target_compile_features(block-allocator-test1 PRIVATE cxx_range_for)

cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include($ENV{COMANCHE_HOME}/mk/common.cmake)

project(blockraid-tests CXX)

find_library(GTEST_LIB NAMES GTEST PATHS /usr/local/lib,/usr/lib)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

# Including extra cmake rules
#include(../../../../mk/clang-dev-tools.cmake)

add_executable(blockraid-test1 test1.cpp)
target_link_libraries(blockraid-test1 ${ASAN_LIB} common numa gtest pthread dl comanche-raid)
target_compile_features(blockraid-test1 PRIVATE cxx_range_for)

add_test(unit_test blockraid-test)

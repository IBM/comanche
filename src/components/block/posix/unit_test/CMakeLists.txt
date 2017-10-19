cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include($ENV{COMANCHE_HOME}/mk/common.cmake)

project(blockposix-tests CXX)

find_library(GTEST_LIB NAMES GTEST PATHS /usr/local/lib,/usr/lib)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

# Including extra cmake rules
#include(../../../../mk/clang-dev-tools.cmake)

add_executable(blockposix-test1 test1.cpp)
target_link_libraries(blockposix-test1 ${ASAN_LIB} common comanche-core gtest pthread comanche-blkposix numa dl)
target_compile_features(blockposix-test1 PRIVATE cxx_range_for)

add_test(unit_test blockposix-test)

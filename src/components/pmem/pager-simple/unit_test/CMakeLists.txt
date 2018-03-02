cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include(./.cmakeinclude)

project(regionpart-tests CXX)

find_library(GTEST_LIB NAMES GTEST PATHS /usr/local/lib,/usr/lib)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

# Including extra cmake rules
#include(../../../../mk/clang-dev-tools.cmake)

add_executable(pagersimple-test1 test1.cpp)
target_link_libraries(pagersimple-test1 ${ASAN_LIB} common numa gtest pthread dl comanche-pagersimple)
target_compile_features(pagersimple-test1 PRIVATE cxx_range_for)

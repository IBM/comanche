cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include(./.cmakeinclude)

project(filestore-tests CXX)

enable_language(CXX C ASM)
set(GCC_COVERAGE_COMPILE_FLAGS "-std=c++11 -g -O2 -fPIC -msse3")

find_library(GTEST_LIB NAMES GTEST PATHS /usr/local/lib,/usr/lib)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

add_executable(filestore-test1 test1.cpp)
target_link_libraries(filestore-test1 ${ASAN_LIB} comanche-core common numa gtest pthread dl)


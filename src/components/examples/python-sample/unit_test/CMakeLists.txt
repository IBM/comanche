cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(pysample-tests CXX)
 
find_library(GTEST_LIB NAMES GTEST PATHS /usr/local/lib,/usr/lib)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -std=c++14 -DCONFIG_DEBUG)

include_directories(../)
include_directories($ENV{COMANCHE_HOME}/src/components)
include_directories($ENV{COMANCHE_HOME}/src/lib/common/include)
include_directories($ENV{COMANCHE_HOME}/src/lib/core/include)

link_directories($ENV{COMANCHE_HOME}/lib)

add_executable(pysample-test1 test1.cpp)
target_link_libraries(pysample-test1 ${ASAN_LIB} common numa gtest pthread dl comanche-allocblock)


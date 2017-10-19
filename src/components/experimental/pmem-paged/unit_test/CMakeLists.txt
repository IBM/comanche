cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

set(GCC_COVERAGE_COMPILE_FLAGS "-std=c++14 -g -O2 -fPIC -msse3")
set(CMAKE_CPP_FLAGS "${CMAKE_CPP_FLAGS} -isystem /usr/local/include")
set(CXXMAKE_C_FLAGS "${CXXMAKE_C_FLAGS} -O2 -g")

project(pmem-paged-tests CXX)

include_directories(../../)
include_directories(../../../components/api/)
include_directories(../../../lib/common/include)
link_directories(/usr/local/lib)
link_directories(../../../lib/common)
link_directories(../)

find_library(GTEST_LIB NAMES GTEST PATHS /usr/local/lib,/usr/lib)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

# Including extra cmake rules
#include(../../../../mk/clang-dev-tools.cmake)

add_executable(pmem-paged-test1 test1.cpp)
target_link_libraries(pmem-paged-test1 numa gtest pthread common dl comanche-pmempaged)
target_compile_features(pmem-paged-test1 PRIVATE cxx_range_for)

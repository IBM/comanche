cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include(../../../mk/common.cmake)

project(pmem-tests CXX)

include_directories(../../)
include_directories(../../../components/)
include_directories(../../../lib/common/include)
include_directories(../../../lib/core/include)
link_directories(/usr/local/lib)
link_directories(../../../lib)

find_library(GTEST_LIB NAMES GTEST PATHS /usr/local/lib,/usr/lib)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

add_executable(pmem-test1 test1.cpp)
target_link_libraries(pmem-test1 common comanche-core numa gtest pthread dl pmem pmemobj)

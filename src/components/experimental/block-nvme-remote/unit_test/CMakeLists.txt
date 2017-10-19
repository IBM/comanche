cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

set(GCC_COVERAGE_COMPILE_FLAGS "-std=c++14 -g -O2 -fPIC -msse3")
set(CMAKE_CPP_FLAGS "${CMAKE_CPP_FLAGS} -isystem /usr/local/include")
set(CXXMAKE_C_FLAGS "${CXXMAKE_C_FLAGS} -O0 -ggdb")

project(blockposix-tests CXX)

# if (DEFINED ENV{COMANCE_HOME})
# else()
#   message("COMANCE_HOME environment variable not defined")
#   exit()
# endif()

include_directories(../../)
include_directories(../src/)
include_directories(../../../../deps/spdk/include/)
include_directories(../../../../deps/dpdk/include/dpdk/)
include_directories(../../../components/api/)
include_directories(../../../lib/common/include)
link_directories(/usr/local/lib)
link_directories(../../../lib/common)
link_directories(../)

find_library(GTEST_LIB NAMES GTEST PATHS /usr/local/lib,/usr/lib)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

# Including extra cmake rules
#include(../../../../mk/clang-dev-tools.cmake)

add_executable(test_client test_client.cpp)
target_link_libraries(test_client numa gtest pthread common dl comanche-blknvme)

add_executable(test_device test_device.cpp)
target_link_libraries(test_device numa gtest pthread common dl comanche-blknvme)

add_executable(test_channel test_channel.cpp)
target_link_libraries(test_channel numa gtest pthread common dl comanche-blknvme rdmacm ibverbs)

add_executable(test_volume_agent test_volume_agent.cpp)
target_link_libraries(test_volume_agent numa gtest pthread common dl comanche-blknvme)

add_executable(test_storage_agent test_storage_agent.cpp)
target_link_libraries(test_storage_agent numa gtest pthread common dl comanche-blknvme )

#target_compile_features(blockposix-test1 PRIVATE cxx_range_for)

add_test(unit_test blockposix-test)

cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include($ENV{COMANCHE_HOME}/mk/common.cmake)

add_subdirectory(./unit_test)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -D_BSD_SOURCE -D_FILE_OFFSET_BITS=64 -MD -MP")
set(DPDK_DIR "$ENV{COMANCHE_HOME}/deps/dpdk/")
set(SPDK_DIR "$ENV{COMANCHE_HOME}/deps/spdk/")

project(comanche-blkposix CXX)

include_directories(${DPDK_DIR}/include/dpdk)
include_directories(${SPDK_DIR}/include)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

file(GLOB SOURCES src/*.cpp)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${ASAN_LIB} common comanche-core dl rt pthread numa rdmacm ibverbs uuid )



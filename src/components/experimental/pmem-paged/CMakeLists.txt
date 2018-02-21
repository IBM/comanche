cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

add_subdirectory(./unit_test)

set(CMAKE_CXX_FLAGS "-std=c++14 -g -O2 -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

project(comanche-pmempaged CXX)

link_directories(../../lib/common)
link_directories(../../../lib)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

file(GLOB SOURCES src/*.cpp)

add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_link_libraries(${PROJECT_NAME} numa pthread common dl rt z)



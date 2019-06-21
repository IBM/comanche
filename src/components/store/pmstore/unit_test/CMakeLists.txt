cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(sample-tests CXX)

enable_language(CXX C ASM)
set (CMAKE_CXX_STANDARD 11)

set(GCC_COVERAGE_COMPILE_FLAGS -g -O2 -fPIC -msse3)

include_directories(${CMAKE_INSTALL_PREFIX}/include)
link_directories(${CMAKE_INSTALL_PREFIX}/lib)
link_directories(${CMAKE_INSTALL_PREFIX}/lib64)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  add_compile_options(${GCC_COVERAGE_COMPILE_FLAGS})
  add_definitions(-DCONFIG_DEBUG)
endif()

add_executable(kvstore-test1 test1.cpp)
target_link_libraries(kvstore-test1 ${ASAN_LIB} common numa gtest pthread dl comanche-pmstore)


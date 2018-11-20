cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(sample-tests CXX)

enable_language(CXX C ASM)
set(GCC_COVERAGE_COMPILE_FLAGS "-std=c++11 -g -O2 -fPIC -msse3")

include_directories(${CMAKE_INSTALL_PREFIX}/include)
link_directories(${CMAKE_INSTALL_PREFIX}/lib)
link_directories(${CMAKE_INSTALL_PREFIX}/lib64)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

add_executable(kvstore-test1 test1.cpp)
target_link_libraries(kvstore-test1 ${ASAN_LIB} common numa gtest pthread dl comanche-pmstore)


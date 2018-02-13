cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


project(fixobd-tests CXX)
 
find_library(GTEST_LIB NAMES GTEST PATHS /usr/local/lib,/usr/lib)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

add_executable(fixobd-test1 test1.cpp)
target_link_libraries(fixobd-test1 ${ASAN_LIB} common comanche-core numa gtest pthread dl)


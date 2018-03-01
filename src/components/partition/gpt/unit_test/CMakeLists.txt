cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include($ENV{COMANCHE_HOME}/mk/common.cmake)

project(gptpart-tests CXX)

find_library(GTEST_LIB NAMES GTEST PATHS /usr/local/lib,/usr/lib)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

add_executable(gptpart-test1 test1.cpp)
target_link_libraries(gptpart-test1 common numa gtest pthread dl comanche-partgpt)
target_compile_features(gptpart-test1 PRIVATE cxx_range_for)

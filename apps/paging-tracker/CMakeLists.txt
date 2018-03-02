cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include(./.cmakeinclude)

project(paging-tracker CXX)

link_directories(/usr/local/lib)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../lib/)
#link_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/)

find_library(GTEST_LIB NAMES GTEST PATHS /usr/local/lib,/usr/lib)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

# Including extra cmake rules
#include(../../../../mk/clang-dev-tools.cmake)

add_executable(paging-tracker src/main.cpp)
target_link_libraries(paging-tracker ${ASAN_LIB} common numa pthread dl z)
target_compile_features(paging-tracker PRIVATE cxx_range_for)

cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


project(core-tests CXX)

include_directories(${CMAKE_INSTALL_PREFIX}/include)
link_directories(${CMAKE_INSTALL_PREFIX}/lib)

find_library(GTEST_LIB NAMES GTEST)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG -mrdrnd)

# Including extra cmake rules
#include(../../../../mk/clang-dev-tools.cmake)

add_executable(core-test1 test1.cpp)
target_link_libraries(core-test1 common comanche-core numa gtest pthread dl)

add_executable(core-test2 test2.cpp)
target_link_libraries(core-test2 common comanche-core numa gtest pthread dl)

add_executable(core-test3 test3.cpp)
target_link_libraries(core-test3 common comanche-core numa gtest pthread dl)

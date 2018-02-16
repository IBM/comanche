cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include(../../../../mk/common.cmake)

project(core-tests CXX)

# if (DEFINED ENV{COMANCE_HOME})
# else()
#   message("COMANCE_HOME environment variable not defined")
#   exit()
# endif()

include_directories(../../)
include_directories(../../../components/)
include_directories(../../../lib/common/include)
include_directories(../../../lib/core/include)
link_directories(/usr/local/lib)
link_directories(../../../../lib)
link_directories(../)

find_library(GTEST_LIB NAMES GTEST PATHS /usr/local/lib,/usr/lib)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

# Including extra cmake rules
#include(../../../../mk/clang-dev-tools.cmake)

add_executable(core-test1 test1.cpp)
target_link_libraries(core-test1 common comanche-core numa gtest pthread dl)


add_executable(core-test2 test2.cpp)
target_link_libraries(core-test2 common comanche-core numa gtest pthread dl)

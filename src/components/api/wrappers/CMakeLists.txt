cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include($ENV{COMANCHE_HOME}/mk/common.cmake)

project(wrappers CXX)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
#add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

add_custom_command(OUTPUT block_itfC_wrap.c COMMAND swig -python block_itfC.h)

set(block_itf_wrapper_SOURCES block_itfC.cc block_itfC_wrap.c)

add_library(comanche-blkwrapper STATIC ${block_itf_wrapper_SOURCES})

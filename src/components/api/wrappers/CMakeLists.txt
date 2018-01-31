cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include($ENV{COMANCHE_HOME}/mk/common.cmake)

project(wrappers CXX)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
#add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

include_directories(/usr/include/python2.7/)

add_custom_command(OUTPUT block_i_wrap.c COMMAND swig -Wall -c++ -python block_i.swg)

set(block_itf_wrapper_SOURCES block_i.cc block_i_wrap.c block_i_wrap.cxx)

add_library(comanche-blkwrapper SHARED ${block_itf_wrapper_SOURCES})

target_link_libraries(comanche-blkwrapper comanche-core)

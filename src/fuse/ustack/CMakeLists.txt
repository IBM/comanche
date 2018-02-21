cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include($ENV{COMANCHE_HOME}/mk/common.cmake)

project(fuse-ustack CXX)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG -D_FILE_OFFSET_BITS=64)

add_executable(fuse-ustack src/fuse_ustack.cpp src/ustack.cpp)
target_link_libraries(fuse-ustack ${ASAN_LIB} common comanche-core numa dl z boost_system boost_program_options pthread fuse)

add_executable(iotest src/iotest.cpp)
target_link_libraries(iotest ${ASAN_LIB} common comanche-core numa dl z boost_system boost_program_options pthread fuse)

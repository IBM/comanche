cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include($ENV{COMANCHE_HOME}/mk/common.cmake)

project(hello C)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG -D_FILE_OFFSET_BITS=64)

add_executable(hello src/hello.c)

target_link_libraries(hello ${ASAN_LIB} common numa dl z boost_system boost_program_options pistache pthread fuse)

cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include(./.cmakeinclude)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_FILE_OFFSET_BITS=64")

add_subdirectory(./unit_test)

project(comanche-fsminix CXX)
enable_language(CXX C ASM)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

file(GLOB_RECURSE SOURCES ./src/*.c*)

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

add_library(comanche-fsminix SHARED ${SOURCES})

target_link_libraries(comanche-fsminix ${ASAN_LIB} common numa pthread dl rt fuse)





cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

# this flag is only for GNU compiler 
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 8.0)
  set(FLAG_DUMP_CLASS "-fdump-class-hierarchy")
endif()

add_subdirectory (sample)
add_subdirectory (python-sample)


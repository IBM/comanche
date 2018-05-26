cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

# don't

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(FLAG_DUMP_CLASS "-fdump-class-hierarchy")
endif()

add_subdirectory (sample)
add_subdirectory (python-sample)


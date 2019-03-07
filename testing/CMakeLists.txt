cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 11)

add_subdirectory (kvstore)
add_subdirectory (integrity)
add_subdirectory (ycsb)

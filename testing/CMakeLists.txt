cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 11)

add_subdirectory (kvstore)
add_subdirectory (integrity)
find_package(MPI)
IF(MPI_FOUND)
add_subdirectory (ycsb)
ENDIF()

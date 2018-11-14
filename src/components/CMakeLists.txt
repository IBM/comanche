cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 11)

add_subdirectory (block)
add_subdirectory (partition)
add_subdirectory (pmem)
add_subdirectory (fs)
add_subdirectory (allocator)
add_subdirectory (store)
add_subdirectory (metadata)
add_subdirectory (net)
add_subdirectory (examples)
add_subdirectory (client)
#add_subdirectory (cluster)


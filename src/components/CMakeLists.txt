cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 11)

execute_process( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )

# x86 only components
if( ${ARCHITECTURE} STREQUAL "x86_64" )
  add_subdirectory (block)
  add_subdirectory (partition)
  add_subdirectory (pmem)
  add_subdirectory (allocator)
endif()

add_subdirectory (store)
add_subdirectory (metadata)
add_subdirectory (net)
add_subdirectory (examples)
add_subdirectory (client)
add_subdirectory (index)
#add_subdirectory (cluster)


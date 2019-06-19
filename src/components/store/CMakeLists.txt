
cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

execute_process( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )

# x86 only components
if( ${ARCHITECTURE} STREQUAL "x86_64" )
  add_subdirectory(hstore)
  add_subdirectory(nvmestore)
  add_subdirectory (append)
  add_subdirectory (log)
  add_subdirectory(pmstore)
  add_subdirectory(dummystore)
endif()

#add_subdirectory (rockstore)
add_subdirectory (filestore)
add_subdirectory(mapstore)

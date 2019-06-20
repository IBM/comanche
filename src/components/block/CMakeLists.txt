cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

execute_process( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )

# x86 only components
if( ${ARCHITECTURE} STREQUAL "x86_64" )
  add_subdirectory (nvme)
  add_subdirectory (unvme)
  add_subdirectory (posix)
  add_subdirectory (raid)
endif()



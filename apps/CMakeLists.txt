cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

execute_process( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )

# x86 only 
if( ${ARCHITECTURE} STREQUAL "x86_64" )
  add_subdirectory (block-perf)
  add_subdirectory (paging-tracker)
endif()

find_package(CUDA)
if(CUDA_FOUND AND CUDA_VERSION GREATER 7.5)
  add_subdirectory(cuda-dma)
  add_subdirectory(gpu-direct)
endif()


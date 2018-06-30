cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 11)

# disable auto-build
if(Nanomsg_FOUND)
  add_subdirectory(ustack)
endif()

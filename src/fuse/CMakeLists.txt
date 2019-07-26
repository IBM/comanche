cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 11)

add_subdirectory(kvfs-simple)
add_subdirectory(hello)
# disable auto-build
find_package(Nanomsg)
if(Nanomsg_FOUND)
  add_subdirectory(ustack)
else()
  message(WARNING "ustack not being build because of nanomsg")

endif()

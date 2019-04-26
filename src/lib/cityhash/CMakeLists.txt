cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

# option to build libraries, on by default
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

ExternalProject_Add(cityhash
  BUILD_IN_SOURCE 1
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cityhash
  CONFIGURE_COMMAND ./configure --with-pic --prefix=${CMAKE_INSTALL_PREFIX}
  BUILD_COMMAND make all
  INSTALL_COMMAND make install
  )


cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

# option to build libraries, on by default
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

# build libfabric
ExternalProject_Add(libfabric
  BUILD_IN_SOURCE 1
  UPDATE_DISCONNECTED 1 # will not update automatically
  GIT_REPOSITORY https://github.com/ofiwg/libfabric.git
  GIT_TAG v1.6.2
  CONFIGURE_COMMAND ./autogen.sh
  COMMAND ./configure --prefix=${CMAKE_INSTALL_PREFIX} --enable-udp=no --enable-mlx=no --enable-sockets=no
  BUILD_COMMAND make
  INSTALL_COMMAND make install
  )

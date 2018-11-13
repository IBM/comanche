cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

# option to build libraies, on by default
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

# build dpdk
ExternalProject_Add(pmdk  
  BUILD_IN_SOURCE 1
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ./bootstrap.sh
  INSTALL_COMMAND make -C pmdk install DESTDIR=${CMAKE_INSTALL_PREFIX} CONFIG_PREFIX=""
  )

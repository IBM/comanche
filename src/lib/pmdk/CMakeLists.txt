cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

# option to build libraies, on by default
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

# build dpdk
ExternalProject_Add(pmdk  
  BUILD_IN_SOURCE 1
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  DOWNLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  CONFIGURE_COMMAND ""
  DOWNLOAD_COMMAND ./bootstrap.sh
  BUILD_COMMAND make -C pmdk
  INSTALL_COMMAND make -C pmdk install prefix=${CMAKE_INSTALL_PREFIX}
  )

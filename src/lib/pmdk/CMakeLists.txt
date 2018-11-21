cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

# option to build libraies, on by default
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

# build dpdk
ExternalProject_Add(pmdk
  DEPENDS libfabric
  BUILD_IN_SOURCE 1
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  DOWNLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  CONFIGURE_COMMAND ""
  DOWNLOAD_COMMAND ./bootstrap.sh
  BUILD_COMMAND PKG_CONFIG_PATH=${CMAKE_INSTALL_PREFIX}/lib/pkgconfig:${CMAKE_INSTALL_PREFIX}/lib64/pkgconfig:${PKG_CONFIG_PATH} make -C pmdk BUILD_RPMEM=n
  INSTALL_COMMAND make -C pmdk install prefix=${CMAKE_INSTALL_PREFIX}
  )

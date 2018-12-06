cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

# option to build libraies, on by default
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

# build pmdk
ExternalProject_Add(pmdk
  DEPENDS libfabric
  DEPENDS ndctl
  BUILD_IN_SOURCE 1
  UPDATE_DISCONNECTED 1
  GIT_REPOSITORY https://github.com/dwaddington/pmdk.git
  GIT_TAG master
  CONFIGURE_COMMAND ""
  BUILD_COMMAND PKG_CONFIG_PATH=${CMAKE_INSTALL_PREFIX}/lib/pkgconfig:${CMAKE_INSTALL_PREFIX}/lib64/pkgconfig make EXTRA_LDFLAGS=-Wl,-rpath=${CMAKE_INSTALL_PREFIX}/lib BUILD_RPMEM=n
  INSTALL_COMMAND EXTRA_LDFLAGS="-Wl,-rpath,${CMAKE_INSTALL_PREFIX}/lib" PKG_CONFIG_PATH=${CMAKE_INSTALL_PREFIX}/lib/pkgconfig:${CMAKE_INSTALL_PREFIX}/lib64/pkgconfig make install prefix=${CMAKE_INSTALL_PREFIX}
  )

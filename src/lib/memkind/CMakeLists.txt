cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

# option to build libraies, on by default
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

ExternalProject_Add(memkind
  BUILD_IN_SOURCE 1
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/memkind
  UPDATE_DISCONNECTED 1
  GIT_REPOSITORY https://github.com/dwaddington/memkind.git
  CONFIGURE_COMMAND ./autogen.sh COMMAND ./configure --prefix=${CMAKE_INSTALL_PREFIX} --sysconfdir=${CMAKE_INSTALL_PREFIX}/etc --libdir=${CMAKE_INSTALL_PREFIX}/lib
  BUILD_COMMAND MAKEOPTS=${COMANCHE_CI_TASK_LIMIT} ./build.sh
  INSTALL_COMMAND make install
  )

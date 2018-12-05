cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

# option to build libraries, on by default
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

ExternalProject_Add(ndctl
  BUILD_IN_SOURCE 1
#  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ndctl
  UPDATE_DISCONNECTED 1
  GIT_REPOSITORY https://github.com/pmem/ndctl.git
  GIT_TAG ndctl-61.y
  CONFIGURE_COMMAND ./autogen.sh COMMAND ./configure --prefix=${CMAKE_INSTALL_PREFIX} --sysconfdir=${CMAKE_INSTALL_PREFIX}/etc --libdir=${CMAKE_INSTALL_PREFIX}/lib
  BUILD_COMMAND make COMMAND make check
  INSTALL_COMMAND install ${CMAKE_CURRENT_BINARY_DIR}/ndctl-prefix/src/ndctl/ndctl/lib/.libs/libndctl.so ${CMAKE_INSTALL_PREFIX}/lib
  COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/include/ndctl/
  COMMAND install ${CMAKE_CURRENT_BINARY_DIR}/ndctl-prefix/src/ndctl/ndctl/libndctl.h ${CMAKE_INSTALL_PREFIX}/include/ndctl/
  COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/include/daxctl/
  COMMAND install ${CMAKE_CURRENT_BINARY_DIR}/ndctl-prefix/src/ndctl/daxctl/libdaxctl.h ${CMAKE_INSTALL_PREFIX}/include/daxctl/
  COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/lib/pkgconfig
  COMMAND install ${CMAKE_CURRENT_BINARY_DIR}/ndctl-prefix/src/ndctl/ndctl/lib/libndctl.pc ${CMAKE_INSTALL_PREFIX}/lib/pkgconfig/
  COMMAND install ${CMAKE_CURRENT_BINARY_DIR}/ndctl-prefix/src/ndctl/daxctl/lib/libdaxctl.pc ${CMAKE_INSTALL_PREFIX}/lib/pkgconfig/
  )

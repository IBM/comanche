cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

# option to build libraies, on by default
option(BUILD_TBB "build tbb library" ON)

include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

add_subdirectory(common)
add_subdirectory(core)

if (BUILD_TBB AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  add_subdirectory(tbb)
endif()

# libkmod is required but missing in fedora
pkg_check_modules(KMOD libkmod)
if(KMOD)
  ExternalProject_Add(ndctl
    BUILD_IN_SOURCE 1
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ndctl
    CONFIGURE_COMMAND ./autogen.sh COMMAND  ./configure --prefix=/usr --sysconfdir=/etc --libdir=/usr/lib64
    BUILD_COMMAND make -j
    INSTALL_COMMAND "" # cannot install, needs sudo privs to do so; install manual from 
    )
endif()


# build pmdk
ExternalProject_Add(pmdk  
  BUILD_IN_SOURCE 1
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pmdk
  CONFIGURE_COMMAND ""
  BUILD_COMMAND make -j
  INSTALL_COMMAND  "" # make install prefix=${CMAKE_INSTALL_PREFIX} don't need to be installed
  )


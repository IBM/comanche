cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

# option to build libraies, on by default
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

add_subdirectory(common)
add_subdirectory(core)

# tbb will be installed into the build-of-root-proj/install/lib/tbb
ExternalProject_Add(tbb
  BUILD_IN_SOURCE 1
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tbb
  CONFIGURE_COMMAND cmake . -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}/tbb
  BUILD_COMMAND make -j2
  INSTALL_COMMAND make install 
  )

ExternalProject_Add(flatbuffers
  BUILD_IN_SOURCE 1
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/flatbuffers
  CONFIGURE_COMMAND cmake . -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}/flatbuffers
  BUILD_COMMAND make -j2
  INSTALL_COMMAND make install 
  )

ExternalProject_Add(nanomsg
  BUILD_IN_SOURCE 1
  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/nanomsg
  CONFIGURE_COMMAND cmake . -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}/nanomsg
  BUILD_COMMAND make -j2
  INSTALL_COMMAND make install 
  )

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
  INSTALL_COMMAND make install prefix=${CMAKE_INSTALL_PREFIX}
  )




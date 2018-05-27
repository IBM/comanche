cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(benchmark)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  add_subdirectory(tbb)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ndctl")
  message("Skipping NDCTL ... already built")
else()
  ExternalProject_Add(ndctl
    GIT_REPOSITORY https://github.com/pmem/ndctl.git
    GIT_TAG v60.3
    #DOWNLOAD_COMMAND git clone -b v60.3 https://github.com/pmem/ndctl.git
    BUILD_IN_SOURCE 1
    UPDATE_DISCONNECTED 1
    DOWNLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ndctl
    CONFIGURE_COMMAND ./autogen.sh COMMAND  ./configure --prefix=/usr --sysconfdir=/etc --libdir=/usr/lib64
    BUILD_COMMAND make -j
    INSTALL_COMMAND "" # cannot install, needs sudo privs to do so; install manual from 
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/pmdk")
  message("Skipping NDCTL ... already built")
else()
  ExternalProject_Add(pmdk  
    GIT_REPOSITORY https://github.com/pmem/pmdk.git
    GIT_TAG stable-1.4
    #DOWNLOAD_COMMAND git clone -b stable-1.4 https://github.com/pmem/pmdk.git
    BUILD_IN_SOURCE 1
    UPDATE_DISCONNECTED 1
    DOWNLOAD_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/pmdk
    CONFIGURE_COMMAND ""
    BUILD_COMMAND make -j
    INSTALL_COMMAND make install prefix=${CMAKE_INSTALL_PREFIX}
    )
endif()


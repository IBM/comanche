cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

execute_process( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )

if( ${ARCHITECTURE} STREQUAL "x86_64" )

  # option to build libraries, on by default
  include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

  # build dpdk
  ExternalProject_Add(dpdk
    BUILD_IN_SOURCE 1
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/bootstrap.sh
    INSTALL_COMMAND ./install.sh ${CMAKE_INSTALL_PREFIX}
    )
  
endif()

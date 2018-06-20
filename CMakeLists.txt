cmake_minimum_required(VERSION 3.5)

# force out-of-tree build
option(FORCE_OTT_BUILD "force out of tree build" ON)

option(USE_RESTRICTED "include build restricted components" OFF)

option(BOOTSTRAP_DEPS "bootstrap to build deps" ON)

if(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR AND FORCE_OTT_BUILD)
    message(FATAL_ERROR "Cannot use in-source build ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}. You should delete any CMakeCache.txt and CMakeFiles and then try out-of-tree build")
endif(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR AND FORCE_OTT_BUILD)

# used for kernel
set(ENV{COMANCHE_HOME} ${CMAKE_CURRENT_SOURCE_DIR})

include(${CMAKE_CURRENT_SOURCE_DIR}/mk/common.cmake)
project(comanche)

# locate FindXXX.cmake
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/mk)
message("CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

# make COMANCHE HOME visible in config.h
set(CONF_COMANCHE_HOME ${CMAKE_CURRENT_SOURCE_DIR})

# override this at cmake time with 'cmake -DCMAKE_INSTALL_PREFIX:PATH=/foo ..'
set(CONF_COMANCHE_INSTALL ${CMAKE_INSTALL_PREFIX})

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/config_comanche.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/config_comanche.h"
    @ONLY)

include_directories(${CMAKE_CURRENT_BINARY_DIR})
install (FILES "${CMAKE_CURRENT_BINARY_DIR}/config_comanche.h"
    DESTINATION include)

# this flag is only for GNU compiler version number smaller than 8 and clang compilers
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 8.0)
  set(FLAG_DUMP_CLASS "-fdump-class-hierarchy")
endif()

# run 'make components' to trigger compoents build
add_custom_target(components
  ${CMAKE_COMMAND} -DBOOTSTRAP_DEPS=off ${CMAKE_CURRENT_SOURCE_DIR}
  COMMAND make all
  )

# default make will only build core and dependencies
if(BOOTSTRAP_DEPS)
  add_subdirectory(src/lib)
  message("\n[IMPORTANT]: first run 'make' to build core and dependencies")
  message("             then run 'make components' to build components")

else(BOOTSTRAP_DEPS)
  set(PMDK_PREFIX ${CONF_COMANCHE_HOME}/src/lib/pmdk)
  #find_package(PMDK REQUIRED)
  set(PMDK_INCLUDE_DIRS  ${PMDK_PREFIX}/src/include)

  add_subdirectory(src)
  add_subdirectory(apps)
  add_subdirectory(testing)

  if (USE_RESTRICTED)
     add_subdirectory(comanche-restricted)
  endif(USE_RESTRICTED)
endif(BOOTSTRAP_DEPS)

# add a target to generate API documentation with Doxygen
find_package(Doxygen)

if(DOXYGEN_FOUND)
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)

add_custom_target(gtags
  GTAGSFORCECPP=1 gtags
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating GTAGS files" VERBATIM
  )

add_custom_target(doc
  ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "Generating API documentation with Doxygen" VERBATIM
  )

endif(DOXYGEN_FOUND)



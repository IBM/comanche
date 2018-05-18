cmake_minimum_required(VERSION 3.5)

# force out-of-tree build
option(FORCE_OTT_BUILD "force out of tree build" ON)

if(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR AND FORCE_OTT_BUILD)
    message(FATAL_ERROR "Cannot use in-source build ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}. You should delete any CMakeCache.txt and CMakeFiles and then try out-of-tree build")
endif(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR AND FORCE_OTT_BUILD)

# used for kernel
set(ENV{COMANCHE_HOME} ${CMAKE_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 11)

include(${CMAKE_SOURCE_DIR}/mk/common.cmake)
project(comanche)

# make COMANCHE HOME visible in config.h
set(CONF_COMANCHE_HOME ${CMAKE_SOURCE_DIR})

# override this at cmake time with 'cmake -DCMAKE_INSTALL_PREFIX:PATH=/foo ..'
set(CONF_COMANCHE_INSTALL ${CMAKE_INSTALL_PREFIX})

configure_file(
    "${CMAKE_SOURCE_DIR}/config_comanche.h.in"
    "${CMAKE_BINARY_DIR}/config_comanche.h"
    @ONLY)
  
include_directories(${CMAKE_BINARY_DIR})
install (FILES "${CMAKE_BINARY_DIR}/config_comanche.h"
    DESTINATION include)


add_subdirectory(src/lib/common)
add_subdirectory(src/lib/core)
add_subdirectory(src/components)
add_subdirectory(apps)
add_subdirectory(src/kernel/modules)


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


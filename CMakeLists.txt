cmake_minimum_required(VERSION 3.5)
include(mk/common.cmake)

if(NOT EXISTS "./src/lib/.cmakeinclude")
  message(SEND_ERROR "You must run (once) autogen.py to generate cmake includes")
  return()
endif()

project(comanche)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/lib/common)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/lib/core)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/components)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/apps)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/modules)


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


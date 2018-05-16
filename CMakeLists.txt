cmake_minimum_required(VERSION 3.5)

  #execute_process(COMMAND python "${CMAKE_CURRENT_LIST_DIR}/autogen.py")
#endif()
set(ENV{COMANCHE_HOME} ${CMAKE_SOURCE_DIR})
include(${CMAKE_SOURCE_DIR}/mk/common.cmake)

project(comanche)


# force out-of-tree build
option(FORCE_OTT_BUILD "force out of tree build" ON)

if(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR AND FORCE_OTT_BUILD)
    message(FATAL_ERROR "cannot use in-souce build ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}, you need delte CMakeCache.txt and CMakeFiles ")
endif(CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR AND FORCE_OTT_BUILD)


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


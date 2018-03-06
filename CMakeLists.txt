cmake_minimum_required(VERSION 3.5)

if(NOT EXISTS "./src/lib/.cmakeinclude")
  message("Generating .cmakeinclude files...")
  execute_process(COMMAND python "${CMAKE_CURRENT_LIST_DIR}/autogen.py")
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


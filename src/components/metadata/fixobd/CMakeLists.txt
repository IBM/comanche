cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

execute_process( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )

project(comanche-mdfixobd)

if( ${ARCHITECTURE} STREQUAL "x86_64" )
  add_subdirectory(./unit_test)
endif()
  
include_directories(${CMAKE_INSTALL_PREFIX}/include)
link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

file(GLOB SOURCES src/*.cpp)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
target_link_libraries(${PROJECT_NAME} common comanche-core pthread numa dl rt tbb)

# set the linkage in the install/lib
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
install (TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)



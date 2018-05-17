cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


add_subdirectory(./unit_test)

project(comanche-zyre CXX)

execute_process(COMMAND git submodule update --init -- lib/zyre
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(lib/zyre/)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)
include_directories(./lib/zyre/include)
link_directories(./lib/zyre/src/.libs/libzyre.a)

file(GLOB SOURCES src/*.cpp)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

target_link_libraries(${PROJECT_NAME} common comanche-core pthread numa dl rt zyre czmq zmq)

# set the linkage in the install/lib
set_target_properties(${PROJECT_NAME} PROPERTIES 
                          INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
install (TARGETS ${PROJECT_NAME}
    LIBRARY 
    DESTINATION lib)



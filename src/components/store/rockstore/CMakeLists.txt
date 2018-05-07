cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include(./.cmakeinclude)

add_subdirectory(./unit_test)

project(comanche-rocksdb CXX)

execute_process(COMMAND git submodule update --init -- rocksdb WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(rocksdblib
  COMMAND make -j PORTABLE=1 static_lib WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/rocksdb/)


enable_language(CXX C ASM)
file(GLOB SOURCES src/*.c*)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}./rocksdb/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}./rocksdb)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
target_link_libraries(${PROJECT_NAME} common comanche-core pthread tbb numa dl rt boost_system boost_filesystem rocksdb)

add_dependencies(${PROJECT_NAME} rocksdblib)

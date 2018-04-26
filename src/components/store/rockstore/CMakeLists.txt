cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include(./.cmakeinclude)

project(comanche-rocksdb CXX)

execute_process(COMMAND git submodule update --init -- rocksdb WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread -fPIC")

add_custom_target(rocksdb_lib
  COMMAND make -j 
  WORKING_DIRECTORY ./rocksdb/)

add_subdirectory(./unit_test)

enable_language(CXX C ASM)
file(GLOB SOURCES src/*.c*)

include_directories(./rocksdb/include)
link_directories(./rocksdb)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -fdump-class-hierarchy -DCONFIG_DEBUG)
add_library(${PROJECT_NAME} SHARED ${SOURCES})
#add_library(rocksdb STATIC IMPORTED)
#set_property(TARGET rocksdb PROPERTY IMPORTED_LOCATION ./rocksdb/librocksdb.a)

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
target_link_libraries(${PROJECT_NAME} common comanche-core pthread tbb numa dl rt boost_system boost_filesystem rocksdb)

#add_dependencies(${PROJECT_NAME} rocksdb_lib)

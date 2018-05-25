cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


add_subdirectory(./unit_test)

project(comanche-rocksdb CXX)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

execute_process(COMMAND git submodule update --init -- rocksdb
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(rocksdblib
  COMMAND make -j ROCKSDB_VALGRIND_RUN=1 EXTRA_LDFLAGS="-fPIC" EXTRA_CFLAGS="-fPIC" EXTRA_CXXFLAGS="-fPIC" static_lib
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/rocksdb/)


enable_language(CXX C ASM)
file(GLOB SOURCES src/*.c*)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/rocksdb/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/rocksdb)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
target_link_libraries(${PROJECT_NAME} common comanche-core tbb numa dl rt boost_system boost_filesystem rocksdb z bz2 pthread snappy lz4)

# set the linkage in the install/lib
set_target_properties(${PROJECT_NAME} PROPERTIES 
                          INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
install (TARGETS ${PROJECT_NAME}
    LIBRARY 
    DESTINATION lib)

add_dependencies(${PROJECT_NAME} rocksdblib)

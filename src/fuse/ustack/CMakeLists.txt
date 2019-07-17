cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include($ENV{COMANCHE_HOME}/mk/common.cmake)

find_package(Flatbuffers REQUIRED)
if(Flatbuffers_FOUND)

project(fuse-ustack CXX)

include_directories(${Nanomsg_INCLUDE_DIRS} ${TBB_INCLUDE_DIRS} ${Flatbuffers_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR}/src)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG -D_FILE_OFFSET_BITS=64)

# if not having this, src won't be in the build dir
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/protocol.fbs"
    "${CMAKE_CURRENT_BINARY_DIR}/src/protocol.fbs"
    @ONLY)

add_custom_command(
  OUTPUT src/protocol_generated.h
  PRE_BUILD
  COMMAND COMMAND ${CMAKE_INSTALL_PREFIX}/bin/flatc --cpp protocol.fbs
  DEPENDS "src/protocol.fbs"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src)

add_custom_target(proto ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/src/protocol_generated.h)

set(USTACK_FS kv_ustack) #blob_ustack is deprecated
foreach(fs ${USTACK_FS})
  # ustack fs built upon blobstore
  add_executable(${fs} src/${fs}.cpp src/ustack.cpp)
  add_dependencies(${fs} proto)

  target_link_libraries(${fs} ${ASAN_LIB} common comanche-core numa dl z ${Nanomsg_LIBRARIES} boost_system boost_program_options pthread ${TBB_LIBRARIES} fuse)
endforeach(fs ${USTACK_FS})

# test for ustack client
add_executable(iotest src/iotest.cpp)
target_link_libraries(iotest ${ASAN_LIB} common comanche-core numa dl z ${Nanomsg_LIBRARIES} boost_system boost_program_options pthread ${TBB_LIBRARIES} fuse)

endif(Flatbuffers_FOUND)

cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include($ENV{COMANCHE_HOME}/mk/common.cmake)

find_package(Flatbuffers REQUIRED)
if(Flatbuffers_FOUND)

project(fuse-ustack CXX)

include_directories(${Nanomsg_INCLUDE_DIRS} ${TBB_INCLUDE_DIRS} ${Flatbuffers_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR}/src)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -D_FILE_OFFSET_BITS=64)

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

# ustack fs built upon blobstore
add_executable(kv_ustack src/kv_ustack.cpp src/ustack.cpp src/ustack_intercepted_io.cpp)
add_dependencies(kv_ustack proto)
# target_compile_definitions(kv_ustack PRIVATE CONFIG_DEBUG=1)

target_link_libraries(kv_ustack ${ASAN_LIB} common comanche-core numa dl z ${Nanomsg_LIBRARIES} boost_system boost_program_options pthread ${TBB_LIBRARIES} fuse)

add_library(ustack_client SHARED src/ustack_client.cpp)
target_link_libraries(ustack_client ${ASAN_LIB} common comanche-core numa dl z ${Nanomsg_LIBRARIES} boost_system boost_program_options pthread ${TBB_LIBRARIES} fuse)

# test for ustack client
#add_executable(iotest src/iotest.cpp)
#target_link_libraries(iotest ${ASAN_LIB} common comanche-core numa dl z ${Nanomsg_LIBRARIES} boost_system boost_program_options pthread ${TBB_LIBRARIES} fuse)

add_subdirectory(unit_test)

endif(Flatbuffers_FOUND)

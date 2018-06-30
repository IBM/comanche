cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include($ENV{COMANCHE_HOME}/mk/common.cmake)

project(fuse-ustack CXX)

include_directories(${Nanomsg_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR}/src)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG -D_FILE_OFFSET_BITS=64)

# if not having this, src won't be in the build dir
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/protocol.fbs"
    "${CMAKE_CURRENT_BINARY_DIR}/src/protocol.fbs"
    @ONLY)

add_custom_command(
  OUTPUT src/protocol_generated.h
  PRE_BUILD
  COMMAND COMMAND ${CONF_COMANCHE_HOME}/src/lib/flatbuffers/flatc --cpp protocol.fbs
  DEPENDS "src/protocol.fbs"
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src)

add_custom_target(proto ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/src/protocol_generated.h)

add_executable(fuse-ustack src/fuse_ustack.cpp src/ustack.cpp)
target_link_libraries(fuse-ustack ${ASAN_LIB} common comanche-core numa dl z ${Nanomsg_LIBRARIES} boost_system boost_program_options pthread ${TBB_LIBRARIES} fuse)

add_executable(iotest src/iotest.cpp)
target_link_libraries(iotest ${ASAN_LIB} common comanche-core numa dl z ${Nanomsg_LIBRARIES} boost_system boost_program_options pthread ${TBB_LIBRARIES} fuse)

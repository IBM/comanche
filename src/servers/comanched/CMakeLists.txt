cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include($ENV{COMANCHE_HOME}/mk/common.cmake)

project(comanched CXX)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

add_custom_command(
  OUTPUT src/protocol_generated.h
  PRE_BUILD
  COMMAND $ENV{COMANCHE_HOME}/deps/flatbuffers/flatc --cpp protocol.fbs
  DEPENDS "src/protocol.fbs"
  WORKING_DIRECTORY ./src)

add_custom_command(
  OUTPUT src/service.pb.cc
  PRE_BUILD
  COMMAND protoc --cpp_out=. -I${CMAKE_CURRENT_BINARY_DIR}/src ${CMAKE_CURRENT_BINARY_DIR}/src/service.proto
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/src/service.proto
  WORKING_DIRECTORY ./src)

#add_custom_target(proto ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/src/protocol_generated.h)
add_custom_target(proto ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/src/service.pb.cc)

add_executable(comanched src/main.cpp src/service.cpp src/service.pb.cc)
target_link_libraries(comanched ${ASAN_LIB} common comanche-core numa pthread dl z boost_system boost_program_options nanomsg protobuf)

add_executable(testcli src/test_client.cpp src/service.pb.cc)
target_link_libraries(testcli ${ASAN_LIB} common comanche-core numa pthread dl z boost_system boost_program_options nanomsg protobuf)

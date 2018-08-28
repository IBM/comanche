cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


project(fabric-tests CXX)

find_library(GTEST_LIB NAMES GTEST PATHS /usr/local/lib,/usr/lib)

include_directories(${CMAKE_INSTALL_PREFIX}/include)
link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

add_executable(fabric-test1 test1.cpp
  eyecatcher.cpp
  mr_lock.cpp
  patience.cpp
  pingpong_client.cpp
  pingpong_server.cpp
  pingpong_server_client_state.cpp
  pingpong_server_n.cpp
  registered_memory.cpp
  registration.cpp
  remote_memory_accessor.cpp
  remote_memory_client.cpp
  remote_memory_client_for_shutdown.cpp
  remote_memory_client_grouped.cpp
  remote_memory_server.cpp
  remote_memory_server_grouped.cpp
  remote_memory_subclient.cpp
  remote_memory_subserver.cpp
  server_connection_and_memory.cpp
  server_connection.cpp
  server_grouped_connection.cpp
  wait_poll.cpp
)

target_link_libraries(fabric-test1 ${ASAN_LIB} common comanche-core fabric gtest pthread dl) # dl

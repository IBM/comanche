cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include_directories(${CONF_COMANCHE_HOME}/testing)

add_executable(test-memory-master test-memory-master.cpp)
target_link_libraries(test-memory-master ${ASAN_LIB} common comanche-core numa pthread gtest dl  nupm)

add_executable(test-memory-slave test-memory-slave.cpp)
target_link_libraries(test-memory-slave  ${ASAN_LIB} common comanche-core numa pthread gtest dl comanche-allocblock profiler nupm)

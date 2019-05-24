cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(nvmestore-tests CXX)

include_directories(${CONF_COMANCHE_HOME}/testing)

link_directories(/usr/local/lib64)
link_directories(/usr/lib/x86_64-linux-gnu)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

add_executable(test-nvmestore test-nvmestore.cpp)
target_link_libraries(test-nvmestore pmem pmemobj ${ASAN_LIB} common comanche-core numa pthread gtest dl comanche-allocblock profiler)

add_executable(test-throughput test-throughput.cpp)
target_link_libraries(test-throughput pmem pmemobj ${ASAN_LIB} common comanche-core numa pthread gtest dl comanche-allocblock profiler)

add_executable(test-integrity test-integrity.cpp)
target_link_libraries(test-integrity pmem pmemobj ${ASAN_LIB} common comanche-core numa pthread gtest dl comanche-allocblock profiler)



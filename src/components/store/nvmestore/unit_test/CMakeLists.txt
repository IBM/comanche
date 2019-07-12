cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(nvmestore-tests CXX)

include_directories(${CONF_COMANCHE_HOME}/testing)

link_directories(/usr/local/lib64)
link_directories(/usr/lib/x86_64-linux-gnu)

#add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

add_executable(test-nvmestore-basic test-nvmestore.cpp)
target_link_libraries(test-nvmestore-basic pmem pmemobj ${ASAN_LIB} common comanche-core numa pthread gtest dl comanche-allocblock profiler)

add_executable(test-nvmestore-throughput test-throughput.cpp)
target_link_libraries(test-nvmestore-throughput pmem pmemobj ${ASAN_LIB} common comanche-core numa pthread gtest dl comanche-allocblock profiler)

add_executable(test-nvmestore-integrity test-integrity.cpp)
target_link_libraries(test-nvmestore-integrity pmem pmemobj ${ASAN_LIB} common comanche-core numa pthread gtest dl comanche-allocblock profiler)


#install(TARGETS test-nvmestore-integrity test-nvmestore-basic test-nvmestore-throughput RUNTIME DESTINATION bin)


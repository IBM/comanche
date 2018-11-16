cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


project(rdma-tests CXX)

link_directories(/usr/local/lib64)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

add_executable(rdma-test1 test1.cpp)
target_link_libraries(rdma-test1 ${ASAN_LIB} common comanche-core numa gtest pthread dl)

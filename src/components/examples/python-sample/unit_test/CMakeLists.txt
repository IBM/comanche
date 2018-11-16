cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


project(pysample-tests CXX)

link_directories(/usr/local/lib64)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -std=c++14 -DCONFIG_DEBUG)

include_directories(../)

add_executable(pysample-test1 test1.cpp)
target_link_libraries(pysample-test1 ${ASAN_LIB} common numa pthread gtest dl comanche-allocblock)


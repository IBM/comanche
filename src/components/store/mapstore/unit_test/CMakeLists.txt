cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


project(mapstore-tests CXX)

set(CMAKE_CXX_STANDARD 11)
set(GCC_COVERAGE_COMPILE_FLAGS "-g -O2 -fPIC")

link_directories(/usr/local/lib64)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

add_executable(mapstore-test1 test1.cpp)
target_link_libraries(mapstore-test1 ${ASAN_LIB} comanche-core common numa gtest pthread dl)


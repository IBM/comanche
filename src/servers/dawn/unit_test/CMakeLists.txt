cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include(${CONF_COMANCHE_HOME}/mk/common.cmake)

project(dawn-test CXX)

include_directories(../include)
link_directories(../)
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)


add_executable(dawn-test ./test_client.cpp)
target_link_libraries(dawn-test ${ASAN_LIB} common comanche-core pthread numa dl rt z)


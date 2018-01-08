cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include($ENV{COMANCHE_HOME}/mk/common.cmake)

project(rest-serv CXX)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

add_executable(rest-serv src/rest_serv.cpp)
target_link_libraries(rest-serv ${ASAN_LIB} common numa dl z boost_system boost_program_options pistache pthread)
target_compile_features(rest-serv PRIVATE cxx_range_for)

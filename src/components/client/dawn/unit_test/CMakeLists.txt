cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


project(dawn-client-test CXX)

enable_language(CXX C ASM)

find_library(GTEST_LIB NAMES GTEST PATHS /usr/local/lib,/usr/lib)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

include_directories(${CMAKE_INSTALL_PREFIX}/include)
link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_executable(dawn-client-test1 test1.cpp)
target_link_libraries(dawn-client-test1 ${ASAN_LIB} common comanche-core numa gtest pthread dl boost_system boost_program_options)

set_target_properties(dawn-client-test1 PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
install(TARGETS dawn-client-test1 RUNTIME DESTINATION bin)

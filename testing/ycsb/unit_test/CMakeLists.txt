cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


project(unit_tests_stats CXX)

enable_language(CXX C ASM)
set(GCC_COVERAGE_COMPILE_FLAGS "-std=c++11 -g -O2 -fPIC -msse3")

link_directories(/usr/local/lib64)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

add_executable(unit_tests_stats test1.cpp)
target_link_libraries(unit_tests_stats ${ASAN_LIB} comanche-core common numa gtest pthread dl)


#project(unit_tests_stopwatch CXX)

#add_executable(unit_tests_stopwatch test_stopwatch.cpp)
#target_link_libraries(unit_tests_stopwatch ${ASAN_LIB} comanche-core common numa gtest pthread dl)


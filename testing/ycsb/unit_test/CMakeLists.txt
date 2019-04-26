cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


project(unit_tests_ycsb CXX)

enable_language(CXX C ASM)
set(GCC_COVERAGE_COMPILE_FLAGS "-std=c++11 -g -O2 -fPIC -msse3")

link_directories(/usr/local/lib64)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

add_executable(unit_tests_ycsb test1.cpp ${SOURCES})
target_link_libraries(unit_tests_ycsb ${ASAN_LIB} comanche-core common numa gtest pthread dl)
#target_include_directories(unit_tests_ycsb PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)


#project(unit_tests_stopwatch CXX)

#add_executable(unit_tests_stopwatch test_stopwatch.cpp)
#target_link_libraries(unit_tests_stopwatch ${ASAN_LIB} comanche-core common numa gtest pthread dl)


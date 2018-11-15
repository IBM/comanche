cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


project(hstore-tests CXX)

enable_language(CXX C ASM)
set(GCC_COVERAGE_COMPILE_FLAGS "-std=c++14 -g -O2 -fPIC -msse3")

find_library(GTEST_LIB NAMES GTEST PATHS /usr/local/lib,/usr/lib)
find_library(PROFILER_LIB NAMES profiler PATHS /usr/local/lib,/usr/lib)
if ("${PROFILER_LIB}" STREQUAL "PROFILER_LIB-NOTFOUND")
set(PROFILER "")
else()
set(PROFILER "profiler")
set(GCC_COVERAGE_COMPILE_FLAGS "${GCC_COVERAGE_COMPILE_FLAGS} -DHAS_PROFILER=1")
endif()


include_directories(${CMAKE_INSTALL_PREFIX}/include)
link_directories(${CMAKE_INSTALL_PREFIX}/lib)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

add_executable(hstore-test1 test1.cpp store_map.cpp)
target_link_libraries(hstore-test1 ${ASAN_LIB} common numa gtest pthread dl comanche-pmstore)
add_executable(hstore-test2 test2.cpp store_map.cpp)
target_link_libraries(hstore-test2 ${ASAN_LIB} common numa gtest pthread dl comanche-pmstore)
add_executable(hstore-test3 test3.cpp store_map.cpp)
target_link_libraries(hstore-test3 ${ASAN_LIB} common numa gtest pthread dl comanche-pmstore ${PROFILER})

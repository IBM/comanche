cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


project(sample-component CXX)

add_subdirectory(./unit_test)

enable_language(CXX C ASM)
set(GCC_COVERAGE_COMPILE_FLAGS "-std=c++11 -g -O2 -fPIC -msse3")

file(GLOB SOURCES src/*.cpp)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -fdump-class-hierarchy -DCONFIG_DEBUG)
add_library(${PROJECT_NAME} SHARED ${SOURCES})


set(GCC_COVERAGE_COMPILE_FLAGS "-std=c++11 -g -O2 -fPIC -msse3")
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
target_link_libraries(${PROJECT_NAME} common comanche-core pthread numa dl rt z)



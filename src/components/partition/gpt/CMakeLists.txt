cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


add_subdirectory(./unit_test)

project(comanche-partgpt CXX)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

file(GLOB SOURCES src/*.cpp)

add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_link_libraries(${PROJECT_NAME} common numa pthread dl rt z)



cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include(./.cmakeinclude)

project(comanche-rocksdb CXX)

add_subdirectory(./unit_test)

enable_language(CXX C ASM)
file(GLOB SOURCES src/*.c*)

include_directories(./rocksdb/include)

# force debug version of PMDK
link_directories(./rocksdb)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -fdump-class-hierarchy -DCONFIG_DEBUG)
add_library(${PROJECT_NAME} SHARED ${SOURCES})


set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
target_link_libraries(${PROJECT_NAME} common comanche-core pthread tbb numa dl rt boost_system boost_filesystem rocksdb)



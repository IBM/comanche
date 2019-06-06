cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(comanche-dummystore CXX)

set (CMAKE_CXX_STANDARD 14)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -O3 -DCONFIG_DEBUG -mavx2)

include_directories(${CMAKE_INSTALL_PREFIX}/include)
link_directories(${CMAKE_INSTALL_PREFIX}/lib)
link_directories(${CMAKE_INSTALL_PREFIX}/lib64)

enable_language(CXX C ASM)
file(GLOB SOURCES src/*.c*)

add_compile_options(-g -Wcast-align -Wcast-qual -Wconversion -Wshadow -Wtype-limits -Wwrite-strings)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

target_link_libraries(${PROJECT_NAME} common comanche-core nupm numa dl rt boost_system pthread cityhash)

# set the linkage in the install/lib
set_target_properties(${PROJECT_NAME} PROPERTIES
  INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/lib64)

install (TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)


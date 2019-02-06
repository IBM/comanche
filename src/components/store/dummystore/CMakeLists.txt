cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(comanche-dummystore CXX)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -O3 -mavx2 -DCONFIG_DEBUG)

include_directories(${CMAKE_INSTALL_PREFIX}/include)
link_directories(${CMAKE_INSTALL_PREFIX}/lib)
link_directories(${CMAKE_INSTALL_PREFIX}/lib64)

enable_language(CXX C ASM)
file(GLOB SOURCES src/*.c*)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

target_link_libraries(${PROJECT_NAME} common comanche-core nupm numa dl rt boost_system pthread cityhash)

# set the linkage in the install/lib
set_target_properties(${PROJECT_NAME} PROPERTIES
  INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/lib64)

install (TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)


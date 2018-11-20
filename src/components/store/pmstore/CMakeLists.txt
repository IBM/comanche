cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(comanche-pmstore CXX)

set (CMAKE_CXX_STANDARD 11)

add_subdirectory(./unit_test)

enable_language(CXX C ASM)
file(GLOB SOURCES src/*.c*)
file(GLOB ALL_SOURCES src/*.[h|c|cpp|cc|hpp] unit_test/*.[h|c|cpp|cc|hpp] )

add_custom_target(format-pmstore
  COMMAND /usr/bin/clang-format
  -style=file
  -i
  ${ALL_SOURCES})

get_property(incdirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
set(sysincludes "-I/usr/include/c++/5 -I/usr/include/x86_64-linux-gnu/c++/5/ -I/usr/include/linux")
# add -I prefix
string(REGEX REPLACE "([^;]+)" "-I\\1" istring "${incdirs}")

add_custom_target(tidy-pmstore
  COMMAND /usr/bin/clang-tidy -header-filter=.* ${ALL_SOURCES} -- ${istring})

include_directories(${CMAKE_INSTALL_PREFIX}/include)
link_directories(${CMAKE_INSTALL_PREFIX}/lib)
link_directories(${CMAKE_INSTALL_PREFIX}/lib64)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -fdump-class-hierarchy -DCONFIG_DEBUG)
endif()
add_library(${PROJECT_NAME} SHARED ${SOURCES})

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
target_link_libraries(${PROJECT_NAME} common comanche-core pthread numa dl rt boost_system boost_filesystem tbb pmem pmemobj pmempool cityhash)

set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/lib64)

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)


cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(nupm CXX C)
enable_language(CXX C)
include(../../../mk/clang-dev-tools.cmake)

find_library(LIB_JEMALLOC libjemalloc.so HINTS ${CMAKE_INSTALL_PREFIX}/lib)
find_library(LIB_NDCTL libndctl.so HINTS ${CMAKE_INSTALL_PREFIX}/lib)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/src")
include_directories(${CMAKE_INSTALL_PREFIX}/include)

link_directories(${CMAKE_INSTALL_PREFIX}/lib)
link_directories(${CMAKE_INSTALL_PREFIX}/lib64)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fopenmp")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

file(GLOB SOURCES src/*.c*)

add_library(nupm SHARED ${SOURCES})

target_link_libraries(${PROJECT_NAME} stdc++fs common numa ${LIB_NDCTL} pmem pmemobj vmem tbb tbbmalloc dl memkind)

add_subdirectory(unit_test)

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)
install(DIRECTORY "src/" DESTINATION "include/nupm" FILES_MATCHING PATTERN "*.h")


# force early install because others depend on this.  this seems a hack. is there
# a better way?
#
add_custom_command(
  TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/include/nupm
  COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/lib
  COMMAND cp ${CMAKE_SOURCE_DIR}/src/lib/nupm/src/*.h ${CMAKE_INSTALL_PREFIX}/include/nupm
  COMMAND cp -r ${CMAKE_SOURCE_DIR}/src/lib/nupm/src/x86_64 ${CMAKE_INSTALL_PREFIX}/include/nupm
  COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/libnupm.so ${CMAKE_INSTALL_PREFIX}/lib
  )


cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(comanche-hstore CXX)

set (CMAKE_CXX_STANDARD 14)

add_subdirectory(./unit_test)

enable_language(CXX)
set(SOURCES src/hstore.cpp src/hop_hash.cpp src/perishable.cpp src/palloc.cpp src/segment_and_bucket.cpp)
# file(GLOB ALL_SOURCES src/*.[h|tcc|cpp] unit_test/*.[h|tcc|cpp] )

#add_custom_target(format-hstore
#  COMMAND /usr/bin/clang-format
#  -style=file
#  -i
#  ${ALL_SOURCES})

add_compile_options(-g -pedantic -Wall -Werror -Wextra -Wcast-align -Wcast-qual -Wconversion -Weffc++ -Wold-style-cast -Wredundant-decls -Wshadow -Wtype-limits -Wunused-parameter -Wwrite-strings)

get_property(incdirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
set(sysincludes "-I/usr/include/c++/5 -I/usr/include/x86_64-linux-gnu/c++/5/ -I/usr/include/linux")
# add -I prefix
string(REGEX REPLACE "([^;]+)" "-I\\1" istring "${incdirs}")

# add_custom_target(tidy-pmstore
#   COMMAND /usr/bin/clang-tidy -header-filter=.* ${ALL_SOURCES} -- ${istring})

include_directories(${CMAKE_INSTALL_PREFIX}/include)
link_directories(${CMAKE_INSTALL_PREFIX}/lib)
include_directories(${CONF_COMANCHE_HOME}/src/lib/pmdk/src/include/)
include_directories(${TBB_INCLUDE_DIRS})

# DO NOT MODIFY

find_library(LIB_PMEM
  NAMES pmem
  HINTS "${CONF_COMANCHE_HOME}/src/lib/pmdk/src/nondebug")

find_library(LIB_PMEM_POOL
  NAMES pmempool
  HINTS "${CONF_COMANCHE_HOME}/src/lib/pmdk/src/nondebug")

find_library(LIB_PMEM_OBJ
  NAMES pmemobj
  HINTS "${CONF_COMANCHE_HOME}/src/lib/pmdk/src/nondebug")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)
endif()
add_library(${PROJECT_NAME} SHARED ${SOURCES})

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
target_link_libraries(${PROJECT_NAME} common comanche-core pthread numa dl rt boost_system boost_filesystem cityhash ${TBB_LIBRARIES} ${LIB_PMEM} ${LIB_PMEM_POOL} ${LIB_PMEM_OBJ})

#set_target_properties(${PROJECT_NAME} PROPERTIES
#  INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${PMDK_PREFIX}/src/nondebug")

#set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib LINKER_LANGUAGE CXX)
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

install(TARGETS ${PROJECT_NAME}
    LIBRARY
    DESTINATION lib)


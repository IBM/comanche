cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include(${CONF_COMANCHE_HOME}/mk/common.cmake)

set(CMAKE_CXX_STANDARD 14)

project(dawn CXX)
include(../../../mk/clang-dev-tools.cmake)

include_directories(${CMAKE_INSTALL_PREFIX}/include)
link_directories(${CMAKE_INSTALL_PREFIX}/lib)
link_directories(${CMAKE_INSTALL_PREFIX}/lib64)

include_directories(include)
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)
#add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DNDEBUG)


set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")

add_executable(dawn src/main.cpp src/shard.cpp src/connection_handler.cpp)

target_link_libraries(dawn ${ASAN_LIB} common comanche-core numa pthread dl pmem boost_program_options) # add profiler

#add_subdirectory(unit_test)

set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/lib64)

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)

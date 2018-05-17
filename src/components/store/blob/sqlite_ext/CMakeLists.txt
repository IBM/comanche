cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


#add_subdirectory(./unit_test)

project(mddb C CXX)

include_directories(../../components/)
link_directories(../../lib/common)
link_directories(../../../lib)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

file(GLOB SOURCES src/*.c*)

set(CMAKE_SHARED_LINKER_FLAGS "-fPIC -shared -Wl,--no-undefined")

add_library(${PROJECT_NAME} SHARED ${SOURCES})
target_link_libraries(${PROJECT_NAME} comanche-core numa pthread common dl rt sqlite3)

# set the linkage in the install/lib
set_target_properties(${PROJECT_NAME} PROPERTIES 
                          INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
install (TARGETS ${PROJECT_NAME}
    LIBRARY 
    DESTINATION lib)



cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


project(sample-component CXX)

set(CMAKE_CXX_STANDARD 11)

add_subdirectory(./unit_test)

enable_language(CXX C ASM)
set(GCC_COVERAGE_COMPILE_FLAGS "-g -O2 -fPIC")

file(GLOB SOURCES src/*.cpp)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} ${FLAG_DUMP_CLASS} -DCONFIG_DEBUG)
add_library(${PROJECT_NAME} SHARED ${SOURCES})


set(GCC_COVERAGE_COMPILE_FLAGS "-g -O2 -fPIC")
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
target_link_libraries(${PROJECT_NAME} common comanche-core pthread numa dl rt z)

# set the linkage in the install/lib
set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
install (TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)



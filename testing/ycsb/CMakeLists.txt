cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

add_subdirectory(unit_test)

#project (ycsb-perf)
#
#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#
#add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)
#
#include_directories(${CMAKE_INSTALL_PREFIX}/include)
#link_directories(${CMAKE_INSTALL_PREFIX}/lib)
#
#enable_language(CXX C ASM)
#file(GLOB SOURCES src/*.c*)
#
#add_executable(ycsb-perf src/*.c*)
#
#target_link_libraries(ycsb-perf comanche-core common numa gtest pthread dl boost_program_options ${TBB_LIBRARIES} profiler boost_system boost_filesystem tbbmalloc)
#
#set_target_properties(${PROJECT_NAME} PROPERTIES 
#  INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
#
#install(TARGETS ycsb-perf RUNTIME DESTINATION bin)

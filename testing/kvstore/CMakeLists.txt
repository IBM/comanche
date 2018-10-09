cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

add_subdirectory(unit_test)

project (kvstore-perf)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

add_executable(kvstore-perf kvstore_perf.cpp)

target_link_libraries(kvstore-perf comanche-core common numa gtest pthread dl boost_program_options ${TBB_LIBRARIES} profiler boost_system boost_filesystem)

set_target_properties(${PROJECT_NAME} PROPERTIES 
  INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

install(TARGETS kvstore-perf RUNTIME DESTINATION bin)

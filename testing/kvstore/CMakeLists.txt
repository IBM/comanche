cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

execute_process( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )

project (kvstore-perf)

add_subdirectory(unit_test)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)
add_compile_options(-g -pedantic -Wall -Werror -Wextra -Wcast-align -Wcast-qual -Wconversion -Weffc++ -Wold-style-cast -Wredundant-decls -Wshadow -Wtype-limits -Wunused-parameter -Wwrite-strings)

add_executable(kvstore-perf kvstore_perf.cpp exp_erase.cpp exp_throughput.cpp experiment.cpp exp_update.cpp program_options.cpp statistics.cpp)

if( ${ARCHITECTURE} STREQUAL "ppc64le" )
  target_link_libraries(kvstore-perf comanche-core common numa gtest pthread dl boost_program_options ${TBB_LIBRARIES} boost_system boost_date_time boost_filesystem tbbmalloc)
else()
  target_link_libraries(kvstore-perf comanche-core common numa gtest pthread dl boost_program_options ${TBB_LIBRARIES} profiler boost_system boost_date_time boost_filesystem tbbmalloc)
endif()

target_compile_options(${PROJECT_NAME} PUBLIC "$<$<CONFIG:Debug>:-O0>")
set_target_properties(${PROJECT_NAME} PROPERTIES 
  INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

install(TARGETS kvstore-perf RUNTIME DESTINATION bin)

cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(block-perf CXX)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

include_directories(${CMAKE_INSTALL_PREFIX}/include)
link_directories(${CMAKE_INSTALL_PREFIX}/lib64)

add_executable(block-perf src/main.cpp)

target_link_libraries(block-perf ${ASAN_LIB} common comanche-core numa pthread dl z boost_system boost_program_options pmem pmemobj pmempool nupm)
target_compile_features(block-perf PRIVATE cxx_range_for)

set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib:${CMAKE_INSTALL_PREFIX}/lib64)
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)

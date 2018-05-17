cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


project(block-perf CXX)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

add_executable(block-perf src/main.cpp)
target_link_libraries(block-perf ${ASAN_LIB} common comanche-core numa pthread dl z boost_system boost_program_options)
target_compile_features(block-perf PRIVATE cxx_range_for)

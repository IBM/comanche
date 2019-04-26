cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


project(blockraid-tests CXX)

link_directories(/usr/local/lib64)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

add_executable(blockraid-test1 test1.cpp)
target_link_libraries(blockraid-test1 ${ASAN_LIB} common numa gtest pthread dl comanche-raid)
target_compile_features(blockraid-test1 PRIVATE cxx_range_for)

add_test(unit_test blockraid-test)

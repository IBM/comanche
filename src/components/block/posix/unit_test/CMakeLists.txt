cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


project(blockposix-tests CXX)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

link_directories(/usr/local/lib64)

add_executable(blockposix-test1 test1.cpp)
target_link_libraries(blockposix-test1 ${ASAN_LIB} common comanche-core gtest pthread comanche-blkposix dl numa)
target_compile_features(blockposix-test1 PRIVATE cxx_range_for)

add_test(unit_test blockposix-test)

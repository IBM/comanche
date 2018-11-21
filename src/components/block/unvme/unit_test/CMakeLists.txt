cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


project(blockunvme-tests CXX)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

link_directories(/usr/local/lib64)

add_executable(blockunvme-test1 test1.cpp)
target_link_libraries(blockunvme-test1 ${ASAN_LIB} common numa gtest pthread dl comanche-blkunvme)
target_compile_features(blockunvme-test1 PRIVATE cxx_range_for)

add_test(unit_test blockunvme-test)

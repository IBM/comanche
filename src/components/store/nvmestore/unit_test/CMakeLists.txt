add_executable(test-nvmestore test-nvmestore.cpp)
add_executable(test-integrity test-integrity.cpp)

include_directories(${CONF_COMANCHE_HOME}/testing)
find_library(GTEST_LIB NAMES GTEST PATHS /usr/local/lib,/usr/lib)
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

target_link_libraries(test-nvmestore pmem pmemobj ${ASAN_LIB}  common comanche-core numa gtest pthread dl comanche-allocblock profiler)
target_link_libraries(test-integrity pmem pmemobj ${ASAN_LIB}  common comanche-core numa gtest pthread dl comanche-allocblock profiler)



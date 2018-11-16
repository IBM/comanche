cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(allocator-blk-aep-tests)

include_directories(./src)
include_directories(${CMAKE_INSTALL_PREFIX}/include)
link_directories(${CMAKE_INSTALL_PREFIX}/lib)
link_directories(/usr/local/lib64)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

add_executable(test-bitmap test-bitmap.cpp ../src/bitmap-tx.cpp)
target_link_libraries(test-bitmap ${ASAN_LIB} common comanche-core numa pthread dl comanche-allocblock pmem pmempool pmemobj tbb profiler gtest )

add_executable(test-blk-alloc-aep test-blk-alloc-aep.cpp)
target_link_libraries(test-blk-alloc-aep ${ASAN_LIB} common comanche-core numa pthread dl comanche-allocblock pmem pmempool pmemobj tbb profiler gtest )


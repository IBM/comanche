add_executable(test-bitmap test-bitmap.cpp ../src/bitmap-tx.cpp)
add_executable(test-blk-alloc-aep test-blk-alloc-aep.cpp)

include_directories(./src)
include_directories(${CMAKE_INSTALL_PREFIX}/include)
link_directories(${CMAKE_INSTALL_PREFIX}/lib)
link_directories(${CMAKE_INSTALL_PREFIX}/lib64)

find_library(GTEST_LIB NAMES GTEST PATHS /usr/local/lib,/usr/lib)
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

target_link_libraries(test-bitmap  ${ASAN_LIB}  common comanche-core numa gtest pthread dl comanche-allocblock pmem pmempool pmemobj)

target_link_libraries(test-blk-alloc-aep ${comanche-blkalloc-aep} pmem pmempool pmemobj ${ASAN_LIB} tbb_static common comanche-core numa gtest pthread dl comanche-allocblock profiler)

#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/check_leaks.sh
#DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

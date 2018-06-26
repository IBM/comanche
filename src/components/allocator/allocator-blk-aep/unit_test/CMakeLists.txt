add_executable(test-bitmap test-bitmap.cpp ../src/bitmap-tx.cpp)
add_executable(test-blk-alloc-aep test-blk-alloc-aep.cpp)

find_library(GTEST_LIB NAMES GTEST PATHS /usr/local/lib,/usr/lib)
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)


target_link_libraries(test-bitmap  ${ASAN_LIB}  common comanche-core numa gtest pthread dl comanche-allocblock  ${LIB_PMEM} ${LIB_PMEM_POOL} ${LIB_PMEM_OBJ})

target_link_libraries(test-blk-alloc-aep ${comanche-blkalloc-aep} ${LIB_PMEM} ${LIB_PMEMOBJ} ${ASAN_LIB}  common comanche-core numa gtest pthread dl comanche-allocblock profiler tbb)

#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/check_leaks.sh
#DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

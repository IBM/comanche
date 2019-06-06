cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project (test_kv)

link_directories(/usr/local/lib64)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

add_executable(test_kvstore test_kvstore.cpp)
add_executable(test_kvindex test_kvindex.cpp)
target_link_libraries(test_kvstore comanche-core common numa gtest pthread dl boost_program_options ${TBB_LIBRARIES} boost_system boost_filesystem)
target_link_libraries(test_kvindex comanche-core common numa gtest pthread dl boost_program_options ${TBB_LIBRARIES} boost_system boost_filesystem)

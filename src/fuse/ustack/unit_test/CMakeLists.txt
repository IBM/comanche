# test for ustack client
include_directories("../src")
add_executable(test-ustack test-ustack.cpp ../src/ustack.cpp ../src/ustack_intercepted_io.cpp)
target_link_libraries(test-ustack ${ASAN_LIB} common comanche-core numa dl z ${Nanomsg_LIBRARIES} boost_system boost_program_options pthread ${TBB_LIBRARIES} fuse profiler)

add_executable(test-preload test_preload.cpp)
target_link_libraries(test-preload dl pthread)

add_executable(test-kvfs test-kvfs.cpp)
target_link_libraries(test-kvfs gtest dl pthread)

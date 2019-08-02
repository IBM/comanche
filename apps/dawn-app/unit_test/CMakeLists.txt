cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(dawn-multithread-test CXX)

enable_language(CXX C ASM)

link_directories(/usr/local/lib64)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

include_directories(${CMAKE_INSTALL_PREFIX}/include)
link_directories(${CMAKE_INSTALL_PREFIX}/lib)

include_directories(${DAWN_APP_SOURCE_DIR}/include)
include_directories($ENV{CUDNN_HOME}/include)
link_directories(${CUDA_SDK_ROOT_DIR}/lib64/stubs)
link_directories($ENV{CUDNN_HOME}/lib64)

add_executable(dawn-multithread-test dawn_multithread_test.cpp)
target_link_libraries(dawn-multithread-test ${ASAN_LIB} common comanche-core numa gtest pthread dl boost_system boost_program_options boost_thread)

cuda_add_executable(dawn-multithread-test-gpu dawn_multithread_test_GPU.cu)
target_link_libraries(dawn-multithread-test-gpu ${ASAN_LIB} common comanche-core numa gtest pthread dl comanche-core common cuda cudart cudadevrt numa cudnn curand boost_system boost_program_options boost_thread)

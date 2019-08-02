cuda_include_directories($ENV{CUDNN_HOME}/include/)

project(conv_bench LANGUAGES CXX)

include_directories(${DAWN_APP_SOURCE_DIR}/include)
include_directories($ENV{CUDNN_HOME}/include)
link_directories(${CUDA_SDK_ROOT_DIR}/lib64/stubs)
link_directories($ENV{CUDNN_HOME}/lib64)

cuda_add_executable(conv_bench conv_bench_kernel.cu)
cuda_add_executable(conv_debug conv_debug.cu)
target_link_libraries(conv_bench comanche-core common cuda cudart cudadevrt numa cudnn curand boost_system boost_program_options) 
target_link_libraries(conv_debug cuda cudart cudadevrt numa cudnn curand) 

cuda_add_executable(conv_bench_file conv_bench_kernel_file.cu)
target_link_libraries(conv_bench_file comanche-core common cuda cudart cudadevrt numa cudnn curand boost_system boost_program_options)

cuda_add_executable(conv_bench_end_to_end conv_bench_kernel_end_to_end.cu)
target_link_libraries(conv_bench_end_to_end comanche-core common cuda cudart cudadevrt numa cudnn curand boost_system boost_program_options)

cuda_add_executable(conv_bench_file_end_to_end conv_bench_kernel_file_end_to_end.cu)
target_link_libraries(conv_bench_file_end_to_end comanche-core common cuda cudart cudadevrt numa cudnn curand boost_system boost_program_options)

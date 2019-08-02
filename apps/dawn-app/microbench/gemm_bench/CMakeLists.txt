cuda_include_directories($ENV{CUDNN_HOME}/include/)

project(gemm_bench LANGUAGES CXX)

include_directories(${DAWN_APP_SOURCE_DIR}/include)
include_directories($ENV{CUDNN_HOME}/include)
link_directories(${CUDA_SDK_ROOT_DIR}/lib64/stubs)
link_directories($ENV{CUDNN_HOME}/lib64)

cuda_add_executable(gemm_bench gemm_bench_kernel.cu)
target_link_libraries(gemm_bench comanche-core common cuda cudart cudadevrt numa cublas cudnn curand boost_system boost_program_options) 

cuda_add_executable(gemm_bench_file gemm_bench_kernel_file.cu)
target_link_libraries(gemm_bench_file comanche-core common cuda cudart cudadevrt numa  cublas cudnn curand boost_system boost_program_options)

#[[
cuda_add_executable(rnn_bench_end_to_end rnn_bench_kernel_end_to_end.cu)
target_link_libraries(rnn_bench_end_to_end comanche-core common cuda cudart cudadevrt numa cublas cudnn curand boost_system boost_program_options)

cuda_add_executable(rnn_bench_file_end_to_end rnn_bench_kernel_file_end_to_end.cu)
target_link_libraries(rnn_bench_file_end_to_end comanche-core common cuda cudart cudadevrt numa cublas cudnn curand boost_system boost_program_options)
]]
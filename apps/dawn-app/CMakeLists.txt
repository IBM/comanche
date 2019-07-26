cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 11)

project(DAWN_APP LANGUAGES CXX)
set(ENV{COMANCHE_HOME} "$ENV{HOME}/comanche")
set(ENV{COMANCHE_INSTALL} "$ENV{HOME}/comanche/build/dist")

IF(DEFINED ENV{CUDNN_HOME})
	message(STATUS "Found cuDNN		: $ENV{CUDNN_HOME}")
ELSE()
	message("CUDNN_HOME is not defined")
ENDIF()


message("COMANCHE_HOME=$ENV{COMANCHE_HOME}")
include_directories($ENV{COMANCHE_INSTALL}/include)
include($ENV{COMANCHE_HOME}/mk/common.cmake)

find_package(CUDA REQUIRED)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda/)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11 -DCONFIG_DEBUG -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70)
set(CUDA_HOST_COMPILATION_CPP ON)
set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_VERBOSE ON)
set(CUDA_64_BIT_DEVICE_CODE ON CACHE STRING "Compile device code in 64 bit mode" FORCE)

message(STATUS "Found CUDA              : ${CUDA_FOUND}")
message(STATUS "Cuda cublas libraries  : ${CUDA_CUBLAS_LIBRARIES}")

cuda_include_directories($ENV{COMANCHE_INSTALL}/include)
cuda_include_directories($ENV{COMANCHE_HOME}/src/components/)
link_directories(/usr/local/cuda/lib64/)
link_directories($ENV{COMANCHE_INSTALL}/lib/)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

add_subdirectory(microbench)
add_subdirectory(unit_test)

set(CMAKE_CXX_STANDARD 11)

set(ENV{COMANCHE_HOME} "$ENV{HOME}/comanche/")
set(ENV{COMANCHE_INSTALL} "$ENV{HOME}/comanche/build/dist/")
set(ENV{GDRCOPY_HOME} "$ENV{HOME}/comanche/build/dist/lib64/")
set(ENV{CUDA_HOME} "/usr/local/cuda")
message("COMANCHE_HOME=$ENV{COMANCHE_HOME}")
include($ENV{COMANCHE_HOME}/mk/common.cmake)

include_directories($ENV{COMANCHE_HOME}/deps/gdrcopy/)
include_directories($ENV{COMANCHE_HOME}/src/components/)
include_directories($ENV{CUDA_HOME}/include)
# build cuda code into static library

link_directories($ENV{COMANCHE_HOME}/deps/gdrcopy/)
link_directories($ENV{COMANCHE_INSTALL}/lib/)
include_directories($ENV{CUDNN_HOME}/include/)


if(USE_KVSTORE)
  list(APPEND Caffe_LINKER_LIBS PUBLIC $ENV{COMANCHE_INSTALL}/lib/libcomanche-core.so $ENV{COMANCHE_INSTALL}/lib/libcommon.so $ENV{GDRCOPY_HOME}/libgdrapi.so $ENV{CUDA_HOME}/lib64/libcudart.so $ENV{CUDA_HOME}/lib64/stubs/libcuda.so)
endif()


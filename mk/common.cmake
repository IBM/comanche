set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE MATCHES Debug)
  message("-- ${CMAKE_CURRENT_SOURCE_DIR} > Debug build.")
elseif(CMAKE_BUILD_TYPE MATCHES Release)
  message("-- ${CMAKE_CURRENT_SOURCE_DIR} > Release build.")
elseif(CMAKE_BUILD_TYPE MATCHES NOOPT)
  message("-- ${CMAKE_CURRENT_SOURCE_DIR} > NOOPT build.")
elseif(CMAKE_BUILD_TYPE MATCHES ASAN)
  message("-- ${CMAKE_CURRENT_SOURCE_DIR} > ASAN build.")
else()
  message("-- ${CMAKE_CURRENT_SOURCE_DIR} > Defaulting to debug build")
  set(CMAKE_BUILD_TYPE Debug)
endif()

#set(GCC_COVERAGE_COMPILE_FLAGS "-fPIC -msse3")
set(CMAKE_CXX_FLAGS_NOOPT "-O0 -g")
set(CMAKE_C_FLAGS_NOOPT "-O0 -g")

set(CMAKE_CXX_FLAGS_DEBUG "-O2 -g")
set(CMAKE_C_FLAGS_DEBUG "-O2 -g")

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_C_FLAGS_RELEASE "-O3")

set(CMAKE_CXX_FLAGS_ASAN "-O0 -ggdb -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_C_FLAGS_ASAN "-O0 -ggdb -fsanitize=address -fno-omit-frame-pointer")

if(${CMAKE_BUILD_TYPE} MATCHES ASAN)
  message("-- Including ASAN runtime...")
  set(ASAN_LIB "asan")
endif()

include_directories($ENV{COMANCHE_HOME}/src/components/)
include_directories($ENV{COMANCHE_HOME}/src/lib/common/include)
include_directories($ENV{COMANCHE_HOME}/src/lib/core/include)
link_directories($ENV{COMANCHE_HOME}/lib)

#set(CMAKE_C_COMPILER "/opt/intel/compilers_and_libraries_2017.2.174/linux/bin/intel64/icc")
#set(CMAKE_CXX_COMPILER "/opt/intel/compilers_and_libraries_2017.2.174/linux/bin/intel64/icc")


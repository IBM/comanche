cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


project(common CXX)
enable_language(CXX C ASM)
  
include(./.cmakeinclude)

set(CMAKE_INCLUDE_CURRENT_DIR true)
include_directories(./include)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -msse4.2)
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)
#add_library(nanomsg SHARED IMPORTED GLOBAL)

add_library(common SHARED cpu.cc rand.cc utils.cc dump_utils.cc str_utils.cc memory.cc crc32.cc city.cc component.cc cycles.cc)

target_compile_features(common PRIVATE cxx_range_for)

# Including extra cmake rules
#include(../../../mk/clang-dev-tools.cmake)




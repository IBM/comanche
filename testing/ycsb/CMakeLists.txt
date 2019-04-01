cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)
project (ycsb-perf CXX)

enable_language(CXX C ASM)
set(GCC_COVERAGE_COMPILE_FLAGS "-std=c++11 -g -O2 -fPIC -msse3 -DAS_USE_LIBEVENT")
#
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})

#
include_directories(${CMAKE_INSTALL_PREFIX}/include)
include_directories(/usr/local/include)
include_directories(/usr/include)
link_directories(${CMAKE_INSTALL_PREFIX}/lib)
link_directories(/usr/local/lib64)
link_directories(/usr/local/lib)
link_directories(/usr/lib)
#
#enable_language(CXX C ASM)
file(GLOB SOURCES src/*.c*)
#
add_executable(ycsb-perf src/ycsb_perf.cpp ${SOURCES})
#

if(MPI_COMPILE_FLAGS)
  set_target_properties(ycsb-perf PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(ycsb-perf PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

target_link_libraries(ycsb-perf comanche-core common numa gtest pthread dl
    boost_program_options ${TBB_LIBRARIES} profiler boost_system
    boost_filesystem tbbmalloc memcached memcachedutil aerospike ssl crypto m z
    hiredis ${MPI_LIBRARIES})
#
set_target_properties(${PROJECT_NAME} PROPERTIES 
  INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

install(TARGETS ycsb-perf RUNTIME DESTINATION bin)

#add_subdirectory(unit_test)

cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

include(./.cmakeinclude)

# We statically compile in the DPDK and SPDK into this module
#
project(comanche-core CXX)

enable_language(CXX C ASM)

set(GCC_COVERAGE_COMPILE_FLAGS "-mrdrnd")
set(DPDK_DIR "$ENV{COMANCHE_HOME}/deps/dpdk/")
set(SPDK_DIR "$ENV{COMANCHE_HOME}/deps/spdk/")

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -mrdrnd)

# DPDK library
find_library(EXTLIB_DPDK libdpdk.a HINTS "${DPDK_DIR}/lib")

if(${EXTLIB_DPDK} STREQUAL "EXTLIB_DPDK-NOTFOUND")
  message(FATAL_ERROR "Couldn't find the 'DPDK' library")
endif()

# SPDK libraries
find_library(EXTLIB_SPDK_ENV_DPDK libspdk_env_dpdk.a HINTS "${SPDK_DIR}/build/lib")
find_library(EXTLIB_SPDK_UTIL libspdk_util.a HINTS "${SPDK_DIR}/build/lib")
find_library(EXTLIB_SPDK_NVME libspdk_nvme.a HINTS "${SPDK_DIR}/build/lib")
find_library(EXTLIB_SPDK_EVENT libspdk_event.a HINTS "${SPDK_DIR}/build/lib")
find_library(EXTLIB_SPDK_LOG libspdk_log.a HINTS "${SPDK_DIR}/build/lib")
find_library(EXTLIB_SPDK_CONF libspdk_conf.a HINTS "${SPDK_DIR}/build/lib")
find_library(EXTLIB_SPDK_TRACE libspdk_trace.a HINTS "${SPDK_DIR}/build/lib")
find_library(EXTLIB_SPDK_IOAT libspdk_ioat.a HINTS "${SPDK_DIR}/build/lib")

if(${EXTLIB_SPDK_NVME} STREQUAL "EXTLIB_SPDK_NVME-NOTFOUND")
  message(FATAL_ERROR "Couldn't find the 'SPDK' library")
endif()

link_directories(../../../deps/dpdk/lib/)
link_directories(../../../lib)

set(CMAKE_INCLUDE_CURRENT_DIR true)
include_directories(./include)
include_directories(./include/core)
include_directories(../../../deps/dpdk/include/dpdk)
include_directories(../../../deps/spdk/include)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

file(GLOB SOURCES src/*.c*)

add_library(comanche-core SHARED ${SOURCES})
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined -Wl,--whole-archive")

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${EXTLIB_DPDK} ${EXTLIB_SPDK_ENV_DPDK} ${EXTLIB_SPDK_NVME}  ${EXTLIB_SPDK_UTIL} ${EXTLIB_SPDK_EVENT} ${EXTLIB_SPDK_TRACE} ${EXTLIB_SPDK_LOG} ${EXTLIB_SPDK_CONF} ${EXTLIB_SPDK_IOAT}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-whole-archive")

target_link_libraries(${PROJECT_NAME} ${ASAN_LIB} common rt pthread numa pcap uuid rdmacm ibverbs dl)

#target_compile_features(comanche-core PRIVATE cxx_range_for)

add_subdirectory(./unit_test)

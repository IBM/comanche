cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


# We statically compile in the DPDK and SPDK into this module
#
project(comanche-core CXX)

enable_language(CXX C ASM)

set(CMAKE_CXX_STANDARD 11)

set(GCC_COVERAGE_COMPILE_FLAGS "-mrdrnd")

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -mrdrnd)

include_directories(${CMAKE_INSTALL_PREFIX}/include)
include_directories(${CMAKE_INSTALL_PREFIX}/include/dpdk)
link_directories(${CMAKE_INSTALL_PREFIX}/lib)

# DPDK library
find_library(EXTLIB_DPDK libdpdk.a)

# SPDK libraries
find_library(EXTLIB_SPDK_ENV_DPDK libspdk_env_dpdk.a)
find_library(EXTLIB_SPDK_UTIL libspdk_util.a)
find_library(EXTLIB_SPDK_NVME libspdk_nvme.a)
find_library(EXTLIB_SPDK_EVENT libspdk_event.a)
find_library(EXTLIB_SPDK_LOG libspdk_log.a)
find_library(EXTLIB_SPDK_CONF libspdk_conf.a)
find_library(EXTLIB_SPDK_TRACE libspdk_trace.a)
find_library(EXTLIB_SPDK_IOAT libspdk_ioat.a)
find_library(EXTLIB_SPDK_RPC libspdk_rpc.a)
find_library(EXTLIB_SPDK_JSONRPC libspdk_jsonrpc.a)
find_library(EXTLIB_SPDK_JSON libspdk_json.a)
find_library(EXTLIB_SPDK_THREAD libspdk_thread.a)

set(CMAKE_INCLUDE_CURRENT_DIR true)
include_directories(./include)
include_directories(./include/core)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

file(GLOB SOURCES src/*.c*)

add_library(comanche-core SHARED ${SOURCES})
set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined -Wl,--whole-archive")

set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${EXTLIB_DPDK} ${EXTLIB_SPDK_ENV_DPDK} ${EXTLIB_SPDK_NVME}  ${EXTLIB_SPDK_UTIL} ${EXTLIB_SPDK_EVENT} ${EXTLIB_SPDK_TRACE} ${EXTLIB_SPDK_LOG} ${EXTLIB_SPDK_CONF} ${EXTLIB_SPDK_IOAT} ${EXTLIB_SPDK_RPC} ${EXTLIB_SPDK_JSONRPC} ${EXTLIB_SPDK_JSON} ${EXTLIB_SPDK_THREAD}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-whole-archive")

target_link_libraries(${PROJECT_NAME} ${ASAN_LIB} common rt pthread numa pcap uuid rdmacm ibverbs dl)

install(TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)
install(DIRECTORY "include/core" DESTINATION include FILES_MATCHING PATTERN "*.h*")

#target_compile_features(comanche-core PRIVATE cxx_range_for)

add_subdirectory(./unit_test)

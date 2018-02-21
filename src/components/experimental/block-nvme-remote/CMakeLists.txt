cmake_minimum_required(VERSION 3.5)

add_subdirectory(./unit_test)

set(GCC_COVERAGE_COMPILE_FLAGS "-g -O1 -fPIC -msse4") # -DCONFIG_DEBUG") beware: bad code -DNDEBUG breaks things
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++14 ")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 -D_BSD_SOURCE -D_FILE_OFFSET_BITS=64 -MD -MP")
set(DPDK_DIR "../../../deps/dpdk/")
set(SPDK_DIR "../../../deps/spdk/")

project(comanche-blknvme CXX)
enable_language(CXX C ASM)

find_library(EXTLIB_COMMON libcommon.a HINTS "../../../lib/")
find_library(EXTLIB_SPDK_ENV_DPDK libspdk_env_dpdk.a HINTS "${SPDK_DIR}/build/lib")
find_library(EXTLIB_SPDK_UTIL libspdk_util.a HINTS "${SPDK_DIR}/build/lib")
find_library(EXTLIB_SPDK_NVME libspdk_nvme.a HINTS "${SPDK_DIR}/build/lib")
find_library(EXTLIB_SPDK_EVENT libspdk_event.a HINTS "${SPDK_DIR}/build/lib")
find_library(EXTLIB_SPDK_LOG libspdk_log.a HINTS "${SPDK_DIR}/build/lib")
find_library(EXTLIB_SPDK_CONF libspdk_conf.a HINTS "${SPDK_DIR}/build/lib")
find_library(EXTLIB_SPDK_TRACE libspdk_trace.a HINTS "${SPDK_DIR}/build/lib")
find_library(EXTLIB_DPDK libdpdk.a HINTS "${DPDK_DIR}/lib")

if(${EXTLIB_SPDK_NVME} STREQUAL "EXTLIB_SPDK_NVME-NOTFOUND")
  message(FATAL_ERROR "Couldn't find the 'SPDK' library")
endif()

if(${EXTLIB_DPDK} STREQUAL "EXTLIB_DPDK-NOTFOUND")
  message(FATAL_ERROR "Couldn't find the 'DPDK' library")
endif()


# find_library(EXT_LIB_DPDK librte_eal.a librte_ethdev.a librte_mbuf.a librte_mempool.a librte_net.a librte_hash.a librte_ring.a librte_sched.a librte_kvargs.a librte_vhost.a librte_pmd_i40e.a librte_pmd_ixgbe.a librte_pmd_e1000.a librte_pmd_ring.a librte_pmd_vhost.a librte_pmd_virtio.a librte_pmd_qede.a librte_pmd_ena.a librte_pmd_cxgbe.a HINTS "../../../deps/dpdk/lib")


include_directories(${DPDK_DIR}/include/dpdk)
include_directories(${SPDK_DIR}/include)
include_directories(./src)
include_directories(./src/lwip/src/include)
include_directories(./src/lwip)
include_directories(../)
include_directories(../../components/api/)

link_directories(../../lib/common)
link_directories(../../../lib)
link_directories(../../../deps/dpdk/lib)
link_directories(../../../deps/spdk/build/lib)

add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

# Including extra cmake rules
#include(../../../mk/clang-dev-tools.cmake)

file(GLOB SOURCES ./src/*.c*)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined -Wl,--whole-archive")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${EXTLIB_SPDK_ENV_DPDK} ${EXTLIB_SPDK_NVME}  ${EXTLIB_SPDK_UTIL} ${EXTLIB_SPDK_EVENT} ${EXTLIB_SPDK_LOG} ${EXTLIB_SPDK_CONF} ${EXTLIB_SPDK_TRACE} ${EXTLIB_COMMON} ")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${EXTLIB_DPDK} -Wl,--no-whole-archive")

target_link_libraries(${PROJECT_NAME} dl rt pthread zmq czmq zyre numa rdmacm ibverbs)

#set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-whole-archive")

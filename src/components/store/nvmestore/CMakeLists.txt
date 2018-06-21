project(comanche-nvmestore CXX)


#add_library(bitmap-tx bitmap-tx.cpp)
include_directories(./src)


include_directories(${PMDK_INCLUDE_DIRS})
include_directories(../../../../src/lib/tbb/include)
include_directories(${PMDK_PREFIX}/src/examples/libpmemobj/hashmap/)
include_directories(${DPDK_DIR}/include/dpdk)


find_library(LIB_PMEM
  NAMES pmem
  HINTS "${CONF_COMANCHE_HOME}/src/lib/pmdk/src/nondebug")

find_library(LIB_PMEM_POOL
  NAMES pmempool
  HINTS "${CONF_COMANCHE_HOME}/src/lib/pmdk/src/nondebug")

find_library(LIB_PMEM_OBJ
  NAMES pmemobj
  HINTS "${CONF_COMANCHE_HOME}/src/lib/pmdk/src/nondebug")


#include_directories(${CONF_COMANCHE_HOME}/comanche-restricted/src/components/pmstore/src/)
file(GLOB SOURCES ./src/*.c*)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

target_link_libraries(${PROJECT_NAME} common comanche-core pthread numa dl rt boost_system boost_filesystem tbb ${LIB_PMEM} ${LIB_PMEM_POOL} ${LIB_PMEM_OBJ} )

set_target_properties(${PROJECT_NAME} PROPERTIES 
  INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;${PMDK_PREFIX}/src/nondebug")

install(TARGETS ${PROJECT_NAME}
    LIBRARY 
    DESTINATION lib)


add_subdirectory(unit_test)

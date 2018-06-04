#.rst:
# FindPMDK
# -------
#
# Finds the PMDK library
# https://cmake.org/cmake/help/v3.11/manual/cmake-developer.7.html#find-modules
# 
# This assumes you have installed PMDK with "sudo make install"
# It will find the pmdk installation(/usr/local) and also the debug version of pmdk
# TODO: an option to select from debug/release(install) version?


# This will define the following variables::
#
#   PMDK_FOUND    - True if the system has the PMDK library
#   PMDK_INCLUDE_DIRS
#   PMDK_LIBRARIES 
#   PMDK_LIBRARIES_DEBUG

find_path(PMDK_INCLUDE_DIR
  NAMES libpmem.h
)

# non-debug libraries, sudo install in pmdk
find_library(LIB_PMEM
  NAMES pmem)

find_library(LIB_PMEMOBJ
  NAMES pmemobj)

find_library(LIB_PMEMPOOL
  NAMES pmempool)

# debug libraries
find_library(LIB_PMEM_DEBUG
  NAMES pmem
  PATHS ${PMDK_PREFIX}/src/debug
  NO_DEFAULT_PATH)

find_library(LIB_PMEMOBJ_DEBUG
  NAMES pmemobj
  PATHS ${PMDK_PREFIX}/src/debug
  NO_DEFAULT_PATH)

find_library(LIB_PMEMPOOL_DEBUG
  NAMES pmempool
  PATHS ${PMDK_PREFIX}/src/debug
  NO_DEFAULT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PMDK
  FOUND_VAR PMDK_FOUND
  REQUIRED_VARS
    PMDK_INCLUDE_DIR
    LIB_PMEM
    LIB_PMEMOBJ
    LIB_PMEMPOOL
    LIB_PMEM_DEBUG
    LIB_PMEMOBJ_DEBUG
    LIB_PMEMPOOL_DEBUG
  VERSION_VAR PMDK_VERSION
)

if(PMDK_FOUND)
  set(PMDK_LIBRARIES ${LIB_PMEM} ${LIB_PMEMOBJ} ${LIB_PMEMPOOL})
  set(PMDK_LIBRARIES_DEBUG ${LIB_PMEM_DEBUG} ${LIB_PMEMOBJ_DEBUG} ${LIB_PMEMPOOL_DEBUG})
  set(PMDK_INCLUDE_DIRS ${PMDK_INCLUDE_DIR})
endif()

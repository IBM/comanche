#.rst:
# FindPMDK
# -------
#
# Finds the PMDK library
# https://cmake.org/cmake/help/v3.11/manual/cmake-developer.7.html#find-modules
# 
# TODO: this can find the pmdk installation with "sudo make install",  an option to select from debug/release(install) version?


# This will define the following variables::
#
#   PMDK_FOUND    - True if the system has the PMDK library
#   PMDK_INCLUDE_DIRS
#   PMDK_LIBRARIES 

find_path(PMDK_INCLUDE_DIR
  NAMES libpmem.h
)


find_library(LIB_PMEM
  NAMES pmem)

find_library(LIB_PMEMOBJ
  NAMES pmemobj)

find_library(LIB_PMEMPOOL
  NAMES pmempool)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PMDK
  FOUND_VAR PMDK_FOUND
  REQUIRED_VARS
    LIB_PMEM
    LIB_PMEMOBJ
    LIB_PMEMPOOL
    PMDK_INCLUDE_DIR
  VERSION_VAR PMDK_VERSION
)

if(PMDK_FOUND)
  set(PMDK_LIBRARIES ${LIB_PMEM} ${LIB_PMEMOBJ} ${LIB_PMEMPOOL})
  set(PMDK_INCLUDE_DIRS ${PMDK_INCLUDE_DIR})
endif()




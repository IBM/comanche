#.rst:
# FindTBB
# -------
#
# Finds the TBB library
# https://cmake.org/cmake/help/v3.11/manual/cmake-developer.7.html#find-modules

# This will define the following variables::
#
#   TBB_FOUND    - True if the system has the TBB library
#   TBB_INCLUDE_DIRS
#   TBB_LIBRARIES 

find_path(TBB_INCLUDE_DIR
  PATHS ${CMAKE_INSTALL_PREFIX}/include
  NAMES tbb/tbb.h
)

# installed libraries
find_library(LIB_TBB
  PATHS ${CMAKE_INSTALL_PREFIX}/lib/
  NAMES tbb)

find_library(LIB_TBB_MALLOC
  PATHS ${CMAKE_INSTALL_PREFIX}/lib/
  NAMES tbbmalloc)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TBB
  FOUND_VAR TBB_FOUND
  REQUIRED_VARS
    TBB_INCLUDE_DIR
    LIB_TBB
  VERSION_VAR TBB_VERSION
)

if(TBB_FOUND)
  set(TBB_LIBRARIES ${LIB_TBB})
  set(TBB_INCLUDE_DIRS ${TBB_INCLUDE_DIR})
endif()

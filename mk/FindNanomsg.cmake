#.rst:
# FindNanomsg
# -------
#
# Finds the Nanomsg library
# https://cmake.org/cmake/help/v3.11/manual/cmake-developer.7.html#find-modules

# This will define the following variables::
#
#   Nanomsg_FOUND    - True if the system has the Nanomsg library
#   Nanomsg_INCLUDE_DIRS
#   Nanomsg_LIBRARIES 

find_path(Nanomsg_INCLUDE_DIR
  PATHS ${CMAKE_INSTALL_PREFIX}/include
  NAMES nanomsg/nn.h
)

# installed libraries
find_library(LIB_Nanomsg
  PATHS ${CMAKE_INSTALL_PREFIX}/lib/
  NAMES nanomsg)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Nanomsg
  FOUND_VAR Nanomsg_FOUND
  REQUIRED_VARS
    Nanomsg_INCLUDE_DIR
    LIB_Nanomsg
  VERSION_VAR Nanomsg_VERSION
)

if(Nanomsg_FOUND)
  set(Nanomsg_LIBRARIES ${LIB_Nanomsg})
  set(Nanomsg_INCLUDE_DIRS ${Nanomsg_INCLUDE_DIR})
endif()

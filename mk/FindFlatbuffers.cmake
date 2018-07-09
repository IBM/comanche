#.rst:
# FindFlatbuffers
# -------
#
# Finds the Flatbuffers library
# https://cmake.org/cmake/help/v3.11/manual/cmake-developer.7.html#find-modules

# This will define the following variables::
#
#   Flatbuffers_FOUND    - True if the system has the Flatbuffers library
#   Flatbuffers_INCLUDE_DIRS
#   Flatbuffers_LIBRARIES 

find_path(Flatbuffers_INCLUDE_DIR
  PATHS ${CMAKE_INSTALL_PREFIX}/include
  NAMES flatbuffers/flatbuffers.h
)

# installed libraries
find_library(LIB_Flatbuffers
  PATHS ${CMAKE_INSTALL_PREFIX}/lib/
  NAMES flatbuffers)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Flatbuffers
  FOUND_VAR Flatbuffers_FOUND
  REQUIRED_VARS
    Flatbuffers_INCLUDE_DIR
    LIB_Flatbuffers
  VERSION_VAR Flatbuffers_VERSION
)

if(Flatbuffers_FOUND)
  set(Flatbuffers_LIBRARIES ${LIB_Flatbuffers})
  set(Flatbuffers_INCLUDE_DIRS ${Flatbuffers_INCLUDE_DIR})
endif()

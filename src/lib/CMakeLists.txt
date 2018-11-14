cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

# option to build libraies, on by default
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

if(BOOTSTRAP_DEPS)

  add_subdirectory(dpdk)
  add_subdirectory(spdk)
  add_subdirectory(pmdk)
  add_subdirectory(libfabric)
  add_subdirectory(flatbuffers)
  add_subdirectory(cityhash)
  add_subdirectory(memkind)
  add_subdirectory(ndctl)
  add_subdirectory(tbb)

  # ExternalProject_Add(nanomsg
  #   BUILD_IN_SOURCE 1
  #   SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/nanomsg
  #   CONFIGURE_COMMAND cmake . -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
  #   BUILD_COMMAND make -j2
  #   INSTALL_COMMAND make install 
  #   )

  find_package(CUDA)
  if(CUDA_FOUND AND CUDA_VERSION GREATER 7.5)
    ExternalProject_Add(gdrcopy  
      BUILD_IN_SOURCE 1
      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/gdrcopy
      CONFIGURE_COMMAND ""
      BUILD_COMMAND PREFIX=${CMAKE_INSTALL_PREFIX} CUDA=${CUDA_TOOLKIT_ROOT_DIR} make all install
      INSTALL_COMMAND PREFIX=${CMAKE_INSTALL_PREFIX} make lib_install
      )
  endif()

else(BOOTSTRAP_DEPS)

  add_subdirectory(common)
  add_subdirectory(core)

endif(BOOTSTRAP_DEPS)






cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

# option to build libraries, on by default
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

if(BOOTSTRAP_DEPS)

  add_subdirectory(dpdk)
  add_subdirectory(spdk)
  add_subdirectory(ndctl)
  add_subdirectory(libfabric)
  add_subdirectory(pmdk)
  add_subdirectory(flatbuffers)
  add_subdirectory(cityhash)
  add_subdirectory(memkind)
  add_subdirectory(tbb)
  add_subdirectory(rapidjson)
  
  find_package(GTest)
  if( NOT GTEST_FOUND) # libgtest-dev in ubuntu 1604 only installs headers
    add_subdirectory(/usr/src/gtest ${CMAKE_CURRENT_BINARY_DIR}/gtest)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/gtest/libgtest.a DESTINATION lib)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/gtest/libgtest_main.a DESTINATION lib)
  endif()


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

  add_subdirectory(nupm)
  add_subdirectory(common)
  add_subdirectory(core)
  add_subdirectory(tds)
  add_subdirectory(dawn-client)
  
endif(BOOTSTRAP_DEPS)




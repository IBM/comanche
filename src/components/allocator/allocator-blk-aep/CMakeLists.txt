project(comanche-blkalloc-aep CXX)

# required by pmdk
include_directories(${CMAKE_INSTALL_PREFIX}/include)
include_directories(./src)
link_directories(${CMAKE_INSTALL_PREFIX}/lib)
link_directories(${CMAKE_INSTALL_PREFIX}/lib64)

file(GLOB SOURCES ./src/*.c*)

add_library(${PROJECT_NAME} SHARED ${SOURCES})

# add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} -DCONFIG_DEBUG)

target_link_libraries(${PROJECT_NAME} common comanche-core pthread numa dl rt boost_system boost_filesystem pmem pmempool pmemobj)

set_target_properties(${PROJECT_NAME} PROPERTIES 
  INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

install(TARGETS ${PROJECT_NAME}
    LIBRARY 
    DESTINATION lib)

add_subdirectory(unit_test)

cmake_minimum_required(VERSION 3.5)

project(kernel-modules)

add_custom_target(xms ALL
  COMMAND make
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/xms)

#add_dependencies(xms)

cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)

project(interfaces CXX)

include(../../../mk/clang-dev-tools.cmake)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include FILES_MATCHING PATTERN "*.h*")

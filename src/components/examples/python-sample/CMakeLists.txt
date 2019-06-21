cmake_minimum_required (VERSION 3.5.1 FATAL_ERROR)


project(pysample-component CXX)

add_subdirectory(./unit_test)

enable_language(CXX C ASM)
set(GCC_COVERAGE_COMPILE_FLAGS "-std=c++11 -g -O2 -fPIC")

FIND_PACKAGE(PythonLibs)
IF(PYTHONLIBS_FOUND)
  INCLUDE_DIRECTORIES("${PYTHON_INCLUDE_DIRS}")
ELSE()
  MESSAGE(FATAL_ERROR "Unable to find PythonLibs.")
ENDIF()

FIND_PACKAGE(Boost)
IF(Boost_FOUND)
  INCLUDE_DIRECTORIES("${Boost_INCLUDE_DIRS}")
  SET(Boost_USE_STATIC_LIBS     OFF)
  SET(Boost_USE_MULTITHREADED    ON)
  SET(Boost_USE_STATIC_RUNTIME     OFF)
  FIND_PACKAGE(Boost  ${BOOST_VERSION} COMPONENTS python) 
ELSEIF(NOT Boost_FOUND)
  MESSAGE(FATAL_ERROR "Unable to find Boost.")
ENDIF()

include_directories(../)

file(GLOB SOURCES src/*.cpp)

find_library(BOOST_PYTHON_PY boost_python-py35)
if(NOT BOOST_PYTHON_PY)
  find_library(BOOST_PYTHON_PY boost_python-py36)
  if(NOT BOOST_PYTHON_PY)
    find_library(BOOST_PYTHON_PY boost_python3) # Fedora
    if(NOT BOOST_PYTHON_PY)      
      message(FATAL_ERROR "boost python not found")
    endif()
  endif()
endif()


add_definitions(${GCC_COVERAGE_COMPILE_FLAGS} ${FLAG_DUMP_CLASS} -DCONFIG_DEBUG)
add_library(${PROJECT_NAME} SHARED ${SOURCES})

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--no-undefined")
target_link_libraries(${PROJECT_NAME} common comanche-core pthread numa dl rt z ${BOOST_PYTHON_PY} ${PYTHON_LIBRARY})

# set the linkage in the install/lib
set_target_properties(${PROJECT_NAME} PROPERTIES 
                          INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)
install (TARGETS ${PROJECT_NAME}
    LIBRARY 
    DESTINATION lib)



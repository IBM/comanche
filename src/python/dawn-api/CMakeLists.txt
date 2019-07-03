cmake_minimum_required(VERSION 3.5) # may work with earlier versions

option(BUILD_PYTHON_SUPPORT "Build Python support" OFF)

find_program(PYTHON "python3.6")

if (PYTHON AND BUILD_PYTHON_SUPPORT)
  
  message("-- Building Python support.")
  set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
  set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
  set(CONFIG_H_IN "${CMAKE_CURRENT_SOURCE_DIR}/src/config.h.in")
  set(CONFIG_H    "${CMAKE_CURRENT_BINARY_DIR}/src/config.h")

  set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py")
  set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")
  set(LIBDIR      "lib.linux-x86_64-3.6")
  
  configure_file(${SETUP_PY_IN} ${SETUP_PY})
  configure_file(${CONFIG_H_IN} ${CONFIG_H})
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/dawn-shell.in ${CMAKE_CURRENT_SOURCE_DIR}/dawn-shell)

  set(DEBUG_COMMAND "${LD_PATH}" "PYTHONPATH=$ENV{PYTHONPATH}:${CMAKE_CURRENT_BINARY_DIR}/build/${LIBDIR}" gdb --args "${PYTHON}" ${PYTHON_FLAGS} -i "${CMAKE_CURRENT_SOURCE_DIR}/normal_start.py")
  set(CHECK_COMMAND "${LD_PATH}" "PYTHONPATH=$ENV{PYTHONPATH}:${CMAKE_CURRENT_BINARY_DIR}/build/${LIBDIR}" "${PYTHON}" ${PYTHON_FLAGS} -i "${CMAKE_CURRENT_SOURCE_DIR}/normal_start.py")
  
  add_custom_command(OUTPUT ${OUTPUT}
    COMMAND ${PYTHON} ${SETUP_PY} build
    COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
    DEPENDS ${DEPS})
                     
  add_custom_target(target ALL DEPENDS ${OUTPUT})
  add_custom_target(debug COMMAND ${DEBUG_COMMAND})
  add_custom_target(check COMMAND ${CHECK_COMMAND})

  install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install --home=${CMAKE_INSTALL_PREFIX})")
  install(FILES dawn-shell DESTINATION ${CMAKE_INSTALL_PREFIX}/bin PERMISSIONS OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE OWNER_READ GROUP_READ WORLD_READ)
  install(FILES dawn-shell-init.py DESTINATION ${CMAKE_INSTALL_PREFIX}/lib PERMISSIONS OWNER_EXECUTE GROUP_EXECUTE WORLD_EXECUTE OWNER_READ GROUP_READ WORLD_READ)
else()
  message("-- NOT building Python support.")
endif()

#set(PYTHON "python3.6-dbg")
#set(LIBNAME "lib.linux-x86_64-3.6-pydebug")
#set(PYTHON "python3.6")
# set(LIBNAME "lib.linux-x86_64-3.6")

# set(IPYTHON "ipython3")
# set(JUPYTER "jupyter")


# # make flags
# add_compile_options(-std=c++11)

# # build vars
# set(PACKAGE_VERSION "0.1")
# set(SETUP_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in")
# set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
# set(DEPS        "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py")
# set(OUTPUT      "${CMAKE_CURRENT_BINARY_DIR}/build")
# #  set(PYTHON_FLAGS "-tt")

# set(LD_PATH "LD_LIBRARY_PATH=${CMAKE_CURRENT_BINARY_DIR}/lib:${CMAKE_CURRENT_BINARY_DIR}/../../../lib:${CMAKE_CURRENT_BINARY_DIR}/../../../lib64:${CMAKE_CURRENT_BINARY_DIR}/../../../comanche/lib:${CMAKE_CURRENT_BINARY_DIR}/../../../comanche/lib64")
# #  set(LD_PATH "LD_LIBRARY_PATH=../../../lib:./lib")

# set(TEST_COMMAND "${LD_PATH}" "PYTHONPATH=$ENV{PYTHONPATH}:${OUTPUT}/${LIBNAME}" "${PYTHON}" ${PYTHON_FLAGS} -i "${CMAKE_CURRENT_SOURCE_DIR}/normal_start.py")

# set(REGRESSION_COMMAND "${LD_PATH}" "PYTHONPATH=$ENV{PYTHONPATH}:${OUTPUT}/${LIBNAME}" "${PYTHON}" ${PYTHON_FLAGS} "${CMAKE_CURRENT_SOURCE_DIR}/regression.py")

# set(HEAPTRACK_COMMAND "${LD_PATH}" "PYTHONPATH=$ENV{PYTHONPATH}:${OUTPUT}/${LIBNAME}" heaptrack "${PYTHON}" ${PYTHON_FLAGS} -i "${CMAKE_CURRENT_SOURCE_DIR}/normal_start.py")

# set(DEBUG_COMMAND "${LD_PATH}" "PYTHONPATH=$ENV{PYTHONPATH}:${OUTPUT}/${LIBNAME}" gdb --args "${PYTHON}" ${PYTHON_FLAGS} -i "${CMAKE_CURRENT_SOURCE_DIR}/normal_start.py")

# set(PYDEBUG_COMMAND "${LD_PATH}" "PYTHONPATH=$ENV{PYTHONPATH}:${OUTPUT}/${LIBNAME}" "${PYTHON}" -m pdb ${PYTHON_FLAGS} -i "${CMAKE_CURRENT_SOURCE_DIR}/normal_start.py")

# set(VALGRIND_COMMAND "${LD_PATH}" "PYTHONPATH=$ENV{PYTHONPATH}:${OUTPUT}/${LIBNAME}" valgrind --tool=memcheck --leak-check=full "${PYTHON}" ${PYTHON_FLAGS} -i "${CMAKE_CURRENT_SOURCE_DIR}/normal_start.py")

# #    set(DEBUG_COMMAND "${LD_PATH}" "PYTHONPATH=$ENV{PYTHONPATH}:${OUTPUT}/${LIBNAME}-pydebug" gdb --args "${PYTHON}" ${PYTHON_FLAGS} -i "${CMAKE_CURRENT_SOURCE_DIR}/normal_start.py")

# # IPython

# set(IPYTHON_FLAGS "--ipython-dir=./")
# set(IPYTHON_COMMAND "${LD_PATH}" "PYTHONPATH=$ENV{PYTHONPATH}:${OUTPUT}/${LIBNAME}" "${IPYTHON}" ${IPYTHON_FLAGS} -i "${CMAKE_CURRENT_SOURCE_DIR}/ipython_start.py")

# set(NOTEBOOK_COMMAND "JUPYTER_CONFIG_DIR=${CMAKE_CURRENT_SOURCE_DIR}/config" "JUPYTER_PATH=${CMAKE_CURRENT_SOURCE_DIR}/notebook" "${LD_PATH}" "PYTHONPATH=$ENV{PYTHONPATH}:${CMAKE_CURRENT_SOURCE_DIR}/notebook:${OUTPUT}/${LIBNAME}" "${JUPYTER}" --certfile=./certs/kite.pem --notebook-dir=./notebook notebook)

# configure_file(${SETUP_PY_IN} ${SETUP_PY})

# add_custom_command(
#   OUTPUT ${OUTPUT}
#   COMMAND ${PYTHON}
#   ARGS setup.py build
#   DEPENDS ${DEPS})

# add_custom_command(
#   OUTPUT ${OUTPUT}/timestamp
#   COMMAND ${PYTHON} setup.py build
#   COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}/timestamp
#   DEPENDS ${DEPS})  

# add_custom_target(bionic ALL DEPENDS ${OUTPUT} ${OUTPUT}/timestamp)
# add_custom_target(check COMMAND ${TEST_COMMAND})
# add_custom_target(regression COMMAND ${REGRESSION_COMMAND})
# add_custom_target(heaptrack COMMAND ${HEAPTRACK_COMMAND})
# add_custom_target(debug COMMAND ${DEBUG_COMMAND})
# add_custom_target(pydebug COMMAND ${PYDEBUG_COMMAND})
# add_custom_target(valgrind COMMAND ${VALGRIND_COMMAND})
# add_custom_target(ipython COMMAND ${IPYTHON_COMMAND})
# add_custom_target(notebook COMMAND ${NOTEBOOK_COMMAND})

# add_custom_target(showipython COMMAND echo ${IPYTHON_COMMAND})
# add_custom_target(showcheck COMMAND echo ${TEST_COMMAND})
# add_custom_target(showdebug COMMAND echo ${DEBUG_COMMAND})

# install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} install)")




#!/usr/bin/env bash

# cmake add_custom_command is stateless, we have to use a script instead

if [ "$#" -ne 1 ]; then
  echo "Illegal number of parameters"
  exit -1
else
  CMAKE_SOURCE_DIR=$1
fi


cmake -DBOOTSTRAP_DEPS=on ${CMAKE_SOURCE_DIR} && \
  make all install &&  \
  cmake -DBOOTSTRAP_DEPS=off ${CMAKE_SOURCE_DIR}

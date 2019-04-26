#!/bin/bash
git submodule update --init --recursive
rm -Rf build
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/dist ..
make bootstrap
echo "Build directory is `pwd`/dist."

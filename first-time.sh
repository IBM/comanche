#!/bin/bash
git submodule update --init --recursive
typeset BUILD=${1:-build}
typeset BUILD_TYPE=${2:-Debug}
rm -Rf ${BUILD}
mkdir -p ${BUILD}
cd ${BUILD}
cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/dist ..
make bootstrap
echo "Build directory is `pwd`/dist."

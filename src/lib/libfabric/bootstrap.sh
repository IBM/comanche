#!/bin/bash
echo "Boot-strapping libfabric ..."

if [ ! -d ./libfabric ] ; then
    git clone -b v1.6.x https://github.com/ofiwg/libfabric.git

fi

# configure and build libfabric
#
cd libfabric ; ./autogen.sh ; ./configure --prefix=$1 --enable-mlx=no ; make


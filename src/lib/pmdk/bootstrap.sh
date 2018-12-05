#!/bin/bash
#
echo "Boot-strapping PMDK (pmem.io) ..."

if [ ! -d ./pmdk ] ; then
    git clone https://github.com/dwaddington/pmdk.git
#    cd pmdk ; git checkout tags/1.4 
fi


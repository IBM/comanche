#!/bin/bash
#
echo "Boot-strapping PMDK (pmem.io) ..."

if [ ! -d ./pmdk ] ; then
    git clone -b stable-1.4 https://github.com/dwaddington/pmdk.git
fi


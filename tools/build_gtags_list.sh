#!/bin/bash
rm -f gtags.files

DIRS="./src"

#include ./src/include ./src/lib/common ./src/lib/comanche ./src/lib/comanche-dd ./src/lib/kivati_client ./src/nvme ./deps/dpdk/ ./deps/spdk/ ./deps/infiniband/"
SUFFIXES="*.h *.cc *.cpp *.c"

for d in $DIRS; do
    for suffix in $SUFFIXES; do
        find $d -name $suffix >> gtags.files
    done
done

#!/bin/bash
rm -f gtags.files

DIRS="./ ./sm ../common/ ../../../deps/zyre/ /usr/local/include"
SUFFIXES="*.h *.cc *.cpp *.c"

for d in $DIRS; do
    for suffix in $SUFFIXES; do
        find $d -name $suffix >> gtags.files
    done
done

gtags .

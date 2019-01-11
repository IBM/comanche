#!/bin/bash
# note you need to run conf-aep-int-devdax2.sh first
#
for i in {0..31}
do
    pmempool info /dev/dax0.$i
done


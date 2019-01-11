#!/bin/bash
# note you need to run conf-aep-int-devdax2.sh first
#
for i in {0..31}
do
    pmempool rm /dev/dax0.$i
    pmempool create obj /dev/dax0.$i
done

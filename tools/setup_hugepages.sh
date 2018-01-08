#!/bin/bash

echo $1 > /proc/sys/vm/nr_hugepages

#sudo mkdir -p /dev/hugepages-1G
#sudo mount -t hugetlbfs -o pagesize=1G nodev /dev/hugepages-1G/

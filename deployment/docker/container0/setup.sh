#!/bin/bash
sudo mkdir -p /dev/hugepages/
sudo mount -a
sudo echo 8000 >  /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages
sudo export SPDK_MEMLIMIT=256


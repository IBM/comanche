#!/bin/bash
make -C ../src/kernel/modules/xms
sudo rmmod xmsmod
sudo insmod ../src/kernel/modules/xms/xmsmod.ko
echo "Inserted XMS kernel module"

  

#!/bin/bash
make -C ./src/kernel/modules/xms
sudo rmmod xmsmod
sudo insmod ./lib/xmsmod.ko
echo "Inserted XMS kernel module"

  

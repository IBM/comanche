#!/bin/bash
export COMANCHE_HOME=`pwd`
echo "Set COMANCHE_HOME variable."
sudo rmmod xmsmod
sudo insmod $COMANCHE_HOME/lib/xmsmod.ko
echo "Inserted XMS kernel module"

  

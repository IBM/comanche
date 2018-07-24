#!/usr/bin/env bash
# Author: Feng Li
# email: fengggli@yahoo.com

# a wrapper scripts for all 


## set pmem
if [ -z `mount\|grep /mnt/huge` ]; then
  echo "ERROR: hugelbfs not mounted at /mnt/huge!setting..."
  sudo mkdir -pv /mnt/huge
  sudo mount -t hugetlbfs nodev /mnt/huge
  echo "you should rerun prepare"
  exit
fi

if [ $PMEM_IS_PMEM_FORCE != 1 ]; then
  echo "ERROR: no xms kernel module found!"
fi

## xms loaded?
if [ -z $(lsmod|grep xms)]; then
  echo "ERROR: no xms kernel module found!"
  exit
fi

## check ulimit -l
if [ x$(ulimit -l) != x"ulimited"]; then
  echo "ERROR: need to set memlock to unlimited at /et/security/limits.conf"
  exit
fi

echo "enviroment setup complete!"

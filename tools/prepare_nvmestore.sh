#!/usr/bin/env bash
# Author: Feng Li
# email: fengggli@yahoo.com

# a wrapper scripts to check and set environment for nvmestore

PINF() {
  echo "[INFO]: $@"
}
PERR() {
  echo "[ERROR]: $@"
}

PINPROGRESS() {
  echo "[NOW SETTING]: $@"
}

PPOSTMSG() {
  echo "[ACTION]: $@"
}

# check current directory
PINF "checking current directory..."
if [ ! -f config_comanche.h.in ]; then
  PERR "you should cd to comanche"
  PPOSTMSG "now exit"
  exit -1
fi
PINF "OK"

# check kernel cmdline
PINF "check kernel cmdline..."
less /proc/cmdline|grep "intel_iommu=on" |grep -q memmap
if [ 0  -ne $? ]; then
  PERR "intel_iommu should be on and reserve space to simulate pmem"
  PINPROGRESS "you should change it upgrade-grub and reboot "
  exit -1
fi
PINF "OK."


# set pmem
PINF "check pmem..."
if [ $(stat -c "%a" /mnt/pmem0/ ) != "777" ]; then
  PERR "pmem not mounted"
  sudo umount /mnt/pmem0
  sudo mkfs.ext4 /dev/pmem0
  sudo mount -o dax /dev/pmem0 /mnt/pmem0
  sudo chmod a+rwx /mnt/pmem0
fi
PINF "OK."

## set huge table
PINF "Checking hugepage."
mount |grep -q /mnt/huge
if [ 0  -ne $? ]; then
  PERR "hugetlbfs not mounted at /mnt/huge!setting..."
  PINPROGRESS "now mounting..." 
  sudo mkdir -pv /mnt/huge
  sudo mount -t hugetlbfs nodev /mnt/huge
  PPOSTMSG "hugetblfs now mounted"
fi
PINF "OK."

## nvme_setup
PINF "Checking VFIO"
ls /dev/vfio -alt |grep -q $USER
if [ 0  -ne $? ]; then
  PERR "vfio is not assigned to current user"
  PINPROGRESS "now reset nvme...step1: unload" 
  sudo ./tools/nvme_setup.sh reset
  PINPROGRESS "now reset nvme...step2: reload" 
  sudo ./tools/nvme_setup.sh
fi
PINF "OK."


## xms loaded and permission ?
PINF "Checking XMS"
lsmod|grep -q xms
if [ 0  -ne $? ]; then
  PERR "no xms kernel module found!"
  PINPROGRESS "now mounting..." 
  sudo load-module.sh
  PPOSTMSG "xms loaded!!"
fi
PINF "OK."

## check ulimit -l
if [ x$(ulimit -l) != x"unlimited" ]; then
  PERR "the memlock limit for current user is low"
  PINPROGRESS "YOU SHOULD: set memlock to unlimited at /et/security/limits.conf, and reboot"
  exit
fi

echo "enviroment setup complete!"

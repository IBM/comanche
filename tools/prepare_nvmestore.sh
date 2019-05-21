#!/usr/bin/env bash
# Author: Feng Li
# email: fengggli@yahoo.com

# a wrapper scripts to check and set environment for nvmestore
if [ "$EUID" -ne 0 ]
  then echo "Please run as root or using sudo"
  exit -1
fi

username=`logname`

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
less /proc/cmdline|grep "intel_iommu=on"  |grep -q memmap
if [ 0  -ne $? ]; then
  PERR "intel_iommu=on needs in kernel cmdline, needs to reserve space to simulate pmem"
  PINPROGRESS "see comanche/src/components/store/nvmestore/HOWTO.md #Kernel Parameters# for more details"
  PINPROGRESS "you should change it upgrade-grub and reboot "
  exit -1
fi
PINF "OK."


# set pmem
PINF "check pmem..."
mkdir -pv /mnt/pmem0
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
ls /dev/vfio -alt
ls /dev/vfio -alt |grep -q $username
if [ 0  -ne $? ]; then
  echo "VFIO not set, run sudo ./tools/attach_to_vfio.sh 11:00.0(using your pci addr), then run prepare_nvmestore.sh again."
  echo "Now exiting... "
  exit -1
fi
PINF "OK."

## xms loaded and permission ?
PINF "Checking XMS"
lsmod|grep -q xms
if [ 0  -ne $? ]; then
  PERR "no xms kernel module found!"
  PINPROGRESS "now mounting..."
  sudo rmmod xmsmod
  sudo insmod ./lib/xmsmod.ko
  ## from xms/reload.sh, need this to allocate EAL mem without sudo
  sudo chmod -R a+rwx /dev/hugepages/
  PPOSTMSG "xms loaded!!"
fi
PINF "OK."

## check ulimit -l
if [ x$(ulimit -l) != x"unlimited" ]; then
  PERR "the memlock limit for current user is low"
  PINPROGRESS "YOU SHOULD: set memlock to unlimited at /etc/security/limits.conf, and reboot"
  PINPROGRESS "see comanche/src/components/store/nvmestore/HOWTO.md # memlock limit# for more details"
  exit
fi

echo "enviroment setup complete!"

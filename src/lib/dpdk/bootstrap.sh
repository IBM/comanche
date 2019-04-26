#!/bin/bash
echo "Boot-strapping DPDK ..."
DPDK_VERSION=dpdk-18.08

if [ ! -d ./$DPDK_VERSION ] ; then
    echo "Downloading DPDK source ...."
    wget https://fast.dpdk.org/rel/$DPDK_VERSION.tar.xz
    tar -xf $DPDK_VERSION.tar.xz
    ln -s ./$DPDK_VERSION dpdk
fi

cp eal_memory.c ./$DPDK_VERSION/lib/librte_eal/linuxapp/eal/eal_memory.c
cp defconfig_x86_64-native-linuxapp-gcc ./$DPDK_VERSION/config
cp build.sh ./$DPDK_VERSION/build.sh

cd ./$DPDK_VERSION/ ; ./build.sh $1

#!/bin/bash

BASE=$PWD

function jumpto
{
    label=$1
    cmd=$(sed -n "/$label:/{:a;n;p;ba};" $0 | grep -v ':$')
    eval "$cmd"
    exit
}

start=${1:-"start"}

jumpto $start

start:

dpdk:
echo "Fetching DPDK v17.08 ..."
rm -f dpdk-17.08*
rm dpdk
wget http://fast.dpdk.org/rel/dpdk-17.08.tar.xz
#wget http://fast.dpdk.org/rel/dpdk-16.11.tar.xz
tar -xvf dpdk-17.08.tar.xz
ln -s dpdk-17.08 dpdk
cd dpdk
mkdir -p usertools # weird hack
cp ../dpdk-extras/build.sh  .
cp ../dpdk-extras/defconfig_x86_64-native-linuxapp-gcc ./config
cp ../dpdk-extras/dpdk-patches-17.08/eal_memory.c ./lib/librte_eal/linuxapp/eal/eal_memory.c
./build.sh
cd $BASE

# jumpto end # TEMPORARY!

spdk:
echo "Cloning SPDK v17.07.1 ..."
git clone https://github.com/spdk/spdk.git
#cd spdk && git checkout tags/v16.12
cd spdk && git checkout fca11f1
./configure --with-dpdk=../dpdk
cp ../spdk-extras/build.sh .
./build.sh
cd $BASE

jumpto end

rocksdb:

echo "Cloning RocksDB..."
git clone https://github.com/facebook/rocksdb.git
cd rocksdb && git checkout tags/v5.1.4

jumpto end

end:

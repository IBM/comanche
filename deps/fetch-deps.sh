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

city:
echo "Cloning CityHash ..."
git clone https://github.com/google/cityhash.git
cd cityhash
./configure && make
sudo make install
cd $BASE

echo "Cloning Google test framework (v1.8.0) ..."
git clone https://github.com/google/googletest.git
cd googletest
git checkout tags/release-1.8.0
cmake . && make
sudo make install
cd $BASE

echo "Cloning Flatbuffers ..."
git clone https://github.com/google/flatbuffers.git
cd flatbuffers
cmake . && make
sudo make install
cd $BASE

# END OF MANDATORT DEPS
jumpto end

nanomsg:
echo "Fetching Nanomsg ..."
git clone https://github.com/nanomsg/nanomsg.git
cd nanomsg ; git checkout tags/1.0.0
cd nanomsg ; ./configure && make ; sudo make install
cd $BASE
jumpto end

protobuf:
echo "Fetching Google protobuf..."
wget https://github.com/google/protobuf/releases/download/v3.0.0/protobuf-cpp-3.0.0.tar.gz
tar -xvf protobuf-cpp-3.0.0.tar.gz
cd protobuf-3.0.0/ ; ./configure ; make ; sudo make install
cd $BASE
jumpto end

messaging:
echo "Cloning libsodium..."
git clone git://github.com/jedisct1/libsodium.git
cd libsodium
./autogen.sh
./configure && make check
sudo make install
sudo ldconfig
cd $BASE
jumpto end

echo "Cloning libzmq..."
git clone git://github.com/zeromq/libzmq.git
cd libzmq
./autogen.sh
./configure && make check
sudo make install
sudo ldconfig
cd $BASE
jumpto end

echo "Cloning CZMQ..."
git clone git://github.com/zeromq/czmq.git
cd czmq
./autogen.sh
./configure && make check
sudo make install
sudo ldconfig
cd $BASE
jumpto end

echo "Cloning libcurve..."
git clone git://github.com/zeromq/libcurve.git
cd libcurve
sh autogen.sh
./autogen.sh
./configure && make check
sudo make install
sudo ldconfig
cd $BASE

# glog:
# echo "Google Glog (v0.3.5) ..."
# git clone git clone https://github.com/google/glog.git
# cd glog
# git checkout tags/v0.3.5
# automake --add-missing && ./configure
# make
# sudo make install
# cd $BASE

zyre:
echo "Cloning Zyre..."
git clone git://github.com/zeromq/zyre.git
cd zyre
./autogen.sh && ./configure && make check
sudo make install
sudo ldconfig
cd $BASE

# moved to apt install
#json:
#echo "Cloning RapidJSON (v1.101)..."
#git clone https://github.com/miloyip/rapidjson.git
#cd rapidjson
#git checkout tags/v1.1.0
#cmake . && make
#sudo make install
#cd $BASE

# end of default install
jumpto end

python:
echo "Fetching Python 3.5 source..."
wget https://www.python.org/ftp/python/3.5.2/Python-3.5.2.tar.xz
tar -xvf Python-3.5.2.tar.xz
cd $BASE

echo "Cloning NumPy source ..."
git clone https://github.com/numpy/numpy.git
cd $BASE

echo "Fetching Armadillio C++ linear algebra library...."
wget http://sourceforge.net/projects/arma/files/armadillo-7.500.0.tar.xz
tar -xvf armadillo-7.500.0.tar.xz
cd $BASE

jumpto end

gpu-bwtest:
echo "Cloning Multi-GPU bandwidth test..."
git clone https://github.com/enfiskutensykkel/multi-gpu-bwtest.git

gpu:
echo "Cloning NVIDIA GDRCOPY ..."
git clone https://github.com/NVIDIA/gdrcopy.git
jumpto end

derecho:

echo "Cloning derecho..."
git clone --recursive https://github.com/Derecho-Project/derecho-unified.git
cd derecho-unified
mkdir build
cd build
cmake .. && make
cd $BASE

jumpto end

rocksdb:

echo "Cloning RocksDB..."
git clone https://github.com/facebook/rocksdb.git
cd rocksdb && git checkout tags/v5.1.4

jumpto end

pmem:
git clone https://github.com/pmem/nvml.git
cd nvml && git checkout tags/1.3
cd $BASE

echo "Cloning PMDK..."
git clone https://github.com/dwaddington/pmdk.git
cd pmdk && make && sudo make install
cd $BASE

jumpto end

cmake:
echo "Cloning cmake..."
wget https://cmake.org/files/v3.6/cmake-3.6.2.tar.gz
tar -zxvf cmake-3.6.2.tar.gz
cd cmake-3.6.2 && ./bootstrap --prefix=/usr && sudo make && sudo make install
jumpto end

end:

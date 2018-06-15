#!/bin/bash

# needed for SPDK and DPDK build
yum install libpcap-devel uuid-devel libuuid libuuid-devel libaio-devel \
    CUnit CUnit-devel librdmacm-devel librdmacm cmake3 numactl-devel python-devel \
    rapidjson-devel gmp-devel mpfr-devel libmpc-devel \
    elfutils-libelf-devel libpcap-devel libuuid-devel libaio-devel boost boost-devel \
    boost-python3 boost-python3-devel doxygen graphviz fuse fuse-devel gperftools gperftools-devel
    
# build GCC 5.4
#curl https://ftp.gnu.org/gnu/gcc/gcc-5.4.0/gcc-5.4.0.tar.bz2 -O
#tar xvfj gcc-5.4.0.tar.bz2
#mkdir gcc-5.4.0-build
#cd gcc-5.4.0-build && ../gcc-5.4.0/configure --enable-languages=c,c++ --disable-multilib && make -j && sudo make install

# apt-get install -y gcc libpciaccess-dev make libcunit1-dev \
#         libaio-dev libssl-dev libibverbs-dev librdmacm-dev libudev-dev uuid \
#         cmake global gdb build-essential\
#         sloccount doxygen synaptic libnuma-dev libaio-dev libcunit1 \
#         libcunit1-dev libboost-system-dev libboost-program-options-dev \
#         libssl-dev g++-multilib fabric libtool-bin autoconf automake \
#         rapidjson-dev libfuse-dev libpcap-dev sqlite3 libsqlite3-dev libomp-dev \
# 	      uuid-dev libtbb-dev libtbb-doc 


# optional
#
# apt-get install emacs elpa-company google-perftools libgoogle-perftools-dev
#

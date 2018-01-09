#!/bin/bash

# needed for SPDK and DPDK build
yum install libpcap-devel uuid-devel libuuid libuuid-devel libaio-devel libcunit1-devel CUnit CUnit-devel librdmacm-devel librdmacm 

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

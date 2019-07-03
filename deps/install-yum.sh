#!/bin/bash

# build-essentials for fedora
yum -y install wget git make automake gcc-c++ openssl-devel sqlite-devel kmod-libs

#yum -y install "kernel-devel-uname-r == $(uname -r)"

# needed for SPDK and DPDK build
yum -y install kmod-devel libudev-devel json-c-devel libpcap-devel uuid-devel libuuid libuuid-devel libaio-devel \
    CUnit CUnit-devel librdmacm-devel librdmacm cmake3 numactl-devel python-devel \
    rapidjson-devel gmp-devel mpfr-devel libmpc-devel \
    elfutils-libelf-devel libpcap-devel libuuid-devel libaio-devel boost boost-devel \
    boost-python3 boost-python3-devel doxygen graphviz fuse fuse-devel gperftools gperftools-devel \
    asciidoc xmlto libtool graphviz gtest gtest-devel pkg-config python3 \
    python3-devel gtest-devel \
    libcurl-devel 

echo "Looking for Python"
find /usr -name "Python.h"

    
    

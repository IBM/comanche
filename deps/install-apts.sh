#!/bin/bash
apt-get install -y gcc libpciaccess-dev make libcunit1-dev pkg-config \
        libaio-dev libssl-dev libibverbs-dev librdmacm-dev libudev-dev uuid uuid-dev\
        cmake global gdb build-essential\
        sloccount doxygen synaptic libnuma-dev libaio-dev libcunit1 \
        libcunit1-dev libboost-system-dev libboost-iostreams-dev libboost-program-options-dev \
        libssl-dev g++-multilib fabric libtool-bin autoconf automake \
        rapidjson-dev libfuse-dev libpcap-dev sqlite3 libsqlite3-dev libomp-dev \
	      libtbb-dev libtbb-doc libboost-python-dev libkmod-dev libjson-c-dev libbz2-dev

# optional
#
# apt-get install emacs elpa-company google-perftools libgoogle-perftools-dev
#

#!/bin/bash
apt-get update
apt-get install -y wget git gcc libpciaccess-dev make libcunit1-dev pkg-config \
        libaio-dev libssl-dev libibverbs-dev librdmacm-dev libudev-dev libuuid1 uuid uuid-dev\
        cmake global gdb build-essential\
        sloccount doxygen synaptic libnuma-dev libaio-dev libcunit1 \
        libcunit1-dev libboost-system-dev libboost-iostreams-dev libboost-program-options-dev \
        libboost-filesystem-dev libboost-date-time-dev \
        libssl-dev g++-multilib fabric libtool-bin autoconf automake \
        rapidjson-dev libfuse-dev libpcap-dev sqlite3 libsqlite3-dev libomp-dev \
	      libboost-python-dev libkmod-dev libjson-c-dev libbz2-dev \
        linux-headers-`uname -r` libelf-dev libsnappy-dev liblz4-dev \
        asciidoc xmlto libtool graphviz \
        google-perftools libgoogle-perftools-dev libgtest-dev \
	libmemcached-dev

# optional
#
# apt-get install emacs elpa-company 
#

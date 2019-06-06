#!/bin/bash
#
# Packages for Ubuntu 18.04 LTS
#
# g++-multilib fabric
apt-get install -y build-essential libnuma-dev libelf-dev libpcap-dev uuid-dev \
        sloccount doxygen synaptic libnuma-dev libaio-dev libcunit1 pkg-config \
        libcunit1-dev libboost-system-dev libboost-iostreams-dev libboost-program-options-dev libboost-filesystem-dev libboost-date-time-dev \
        libssl-dev libtool-bin autoconf automake libibverbs-dev librdmacm-dev \
        rapidjson-dev libfuse-dev libpcap-dev sqlite3 libsqlite3-dev libomp-dev \
        libboost-python-dev libkmod-dev libjson-c-dev libbz2-dev \
        linux-headers-`uname -r` libelf-dev libsnappy-dev liblz4-dev \
        asciidoc xmlto libtool libgtest-dev google-perftools libgoogle-perftools-dev

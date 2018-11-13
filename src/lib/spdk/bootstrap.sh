#!/bin/bash
#
# version should correspond to DPDK
#
echo "Boot-strapping SPDK ..."

if [ ! -d ./spdk ] ; then
    git clone -b v18.10 https://github.com/spdk/spdk.git
fi

cd spdk ; ./configure --with-dpdk=$1/share/dpdk/x86_64-native-linuxapp-gcc/ --without-vhost --without-virtio
make DPDK_DIR=$1/share/dpdk/x86_64-native-linuxapp-gcc/ CONFIG_RDMA=y

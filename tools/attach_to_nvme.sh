#!/bin/bash

device=$1

# detach from vfio driver if needed
#
if [ -e /sys/bus/pci/drivers/vfio-pci/0000:$device ] ; then
    echo "Detaching $device from vfio-pci kernel driver.."
    echo 0000:$device > /sys/bus/pci/drivers/vfio-pci/unbind
else
    echo "Device not attached to vfio driver"
fi


# attach to nvme driver
echo 0000:$device > /sys/bus/pci/drivers/nvme/bind


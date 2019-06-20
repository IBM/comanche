#!/bin/bash

device=$1

# detach from kernel driver if needed
#
if [ -e /sys/bus/pci/drivers/nvme/0000:$device ] ; then
    echo "Detaching $device from nvme kernel driver.."
    echo 0000:$device > /sys/bus/pci/drivers/nvme/unbind
else
    echo "Device not attached to nvme kernel driver"
fi


# attach to vfio driver
modprobe vfio-pci
ven_dev_id=$(lspci -n -s $device | cut -d' ' -f3 | sed 's/:/ /')
echo Device: $ven_dev_id
echo "$ven_dev_id" > "/sys/bus/pci/drivers/vfio-pci/new_id" 2> /dev/null || true
echo "$device" > "/sys/bus/pci/drivers/vfio-pci/bind" 2> /dev/null || true
echo "$device bound to vfio-pci driver."

username=`logname`
iommu_group=$(basename $(readlink -f /sys/bus/pci/devices/0000:$device/iommu_group))
if [ -e "/dev/vfio/$iommu_group" ]; then
    chown "$username" "/dev/vfio/$iommu_group"
fi

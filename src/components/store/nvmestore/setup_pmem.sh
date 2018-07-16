#!/usr/bin/env bash

sudo umount /mnt/pmem0
sudo mkfs.ext4 /dev/pmem0
sudo mount -o dax /dev/pmem0 /mnt/pmem0
sudo chmod a+rwx /mnt/pmem0



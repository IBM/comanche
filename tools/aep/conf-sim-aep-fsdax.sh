#!/bin/bash
sudo mkfs.xfs -f -d su=2m,sw=1 /dev/pmem0
sudo mkdir -p /mnt/pmem0
sudo mount -o dax /dev/pmem0 /mnt/pmem0
sudo chmod a+rwx /mnt/pmem0
sudo xfs_io -c "extsize 2m" /mnt/pmem0/
echo "Done!"

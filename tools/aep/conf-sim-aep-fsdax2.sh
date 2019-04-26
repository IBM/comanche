#!/bin/bash

for i in {0..1}
do
    sudo mkfs.xfs -f -d su=2m,sw=1 /dev/pmem$i
    sudo mkdir -p /mnt/pmem$i
    sudo mount -o dax /dev/pmem$i /mnt/pmem$i
    sudo chmod a+rwx /mnt/pmem$i
    sudo xfs_io -c "extsize 2m" /mnt/pmem$i/
done
echo "Done!"

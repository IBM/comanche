#!/bin/bash
FSDAX_PART_SIZE=6G
ALIGNMENT=1G

if [ "$1" == "hwinit" ]
then
   read -p "Are you sure you want to configure HW and erase AEP namespaces?" -n 1 -r
   echo
   if [[ $REPLY =~ ^[Yy]$ ]]
   then
       echo "Creating goals..."
       sudo ndctl destroy-namespace all --force
       sudo ipmctl delete -goal
       sudo ipmctl create -goal PersistentMemoryType=AppDirect
       echo "Goal configuration complete. Reboot and then run this script with first argument \"part2\""
       exit
   fi
else
    sudo umount /mnt/pmem*
    echo "Destroying existing namespaces."
    sudo ndctl destroy-namespace all --force
    
    REGIONS=`ndctl list -R | egrep -oh 'region[0-9]+' | egrep -oh '[0-9]+'`
    echo "Creating namespaces... this will take a while!"    
    for r in $REGIONS
    do
        # create devdax partition
        echo "Creating base fsdax partition..."
        sudo ndctl create-namespace -s $FSDAX_PART_SIZE -m fsdax --align $ALIGNMENT -r $r

        # rest of space is fsdax
        echo "Creating devdax partition..."
	      sudo ndctl create-namespace -m devdax --align $ALIGNMENT -r $r
    done

    for r in $REGIONS
    do        
        # create filesystem and format partition
        echo "Formatting fsdax partitions..."
        sudo mkfs.xfs -f /dev/pmem$r
        sudo mount /dev/pmem$r /mnt/pmem$r
        sudo chmod a+rwx /mnt/pmem$r
    done
    
    sudo chmod a+rwx /dev/dax*
fi


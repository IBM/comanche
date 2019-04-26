#!/bin/bash
if [ "$1" == "part2" ]
then
    REGIONS=`ndctl list -R | egrep -oh 'region[0-9]+'`
    echo "Creating namespaces... this will take a while!"
    for R in $REGIONS
    do
        echo ndctl create-namespace -m devdax -r $R --align 1G
        sudo ndctl create-namespace -m devdax -r $R --align 1G
    done
    sudo chmod a+rwx /dev/dax*
    ls -l /dev/dax*
else
    read -p "Are you sure you want to erase AEP namespaces?" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        echo "Creating goals..."
        sudo ndctl destroy-namespace all --force
        sudo ipmctl delete -goal
        sudo ipmctl create -goal PersistentMemoryType=AppDirectNotInterleaved
        echo "Goal configuration complete. Reboot and then run this script with first argument \"part2\""
        exit
    fi
fi


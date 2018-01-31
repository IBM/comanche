#!/bin/bash

read -p "Are you sure you want to blitz $1 ?" -n 1 -r
echo    # (optional) move to a new line
if [[ $REPLY =~ ^[Yy]$ ]]
then
    
# to create the partitions programatically (rather than manually)
# we're going to simulate the manual input to fdisk
# The sed script strips off all the comments so that we can 
# document what we're doing in-line with the actual commands
# Note that a blank line (commented as "defualt" will send a empty
# line terminated with a newline to take the fdisk default.
sed -e 's/\s*\([\+0-9a-zA-Z]*\).*/\1/' << EOF | fdisk $1
  g # create GUID partition table
  n # new partition
  1 # partition number 1
    # default - start at beginning of disk 
  +2G # create 2G parttion
  t # set type
  21 # Linux server data
  n # new partition
  2  # partion number 2
    # default, start immediately after preceding partition
    # default, extend partition to end of disk
  t # set type
  2 # partition 2
  21 # Linux server data
  w # write the partition table
  p
  q # and we're done
EOF

fi

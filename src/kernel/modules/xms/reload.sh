#!/bin/bash
sudo rmmod xmsmod
sudo insmod xmsmod.ko
sudo chmod -R a+rwx /dev/hugepages/

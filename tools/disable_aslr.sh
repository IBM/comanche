#!/bin/bash

echo 0 | sudo tee /proc/sys/kernel/randomize_va_space


# renable with
# echo 2 | sudo tee /proc/sys/kernel/randomize_va_space

# permanent solution
#
# This won't survive a reboot, so you'll have to configure this in
# sysctl. Add a file /etc/sysctl.d/01-disable-aslr.conf containing:
#
# kernel.randomize_va_space = 0


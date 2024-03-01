#!/bin/bash

# Clear PageCache
sync
echo 1 > /proc/sys/vm/drop_caches

# Clear dentries and inodes
sync
echo 2 > /proc/sys/vm/drop_caches

# Clear PageCache, dentries, and inodes
sync
echo 3 > /proc/sys/vm/drop_caches

echo "Caches cleared successfully."

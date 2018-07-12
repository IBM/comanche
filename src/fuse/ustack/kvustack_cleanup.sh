#!/usr/bin/env bash

# Author: Feng Li
# email: fengggli@yahoo.com

fusermount -u  ./mymount
rm fifo* -rf
rm /mnt/pmem0/* -rf

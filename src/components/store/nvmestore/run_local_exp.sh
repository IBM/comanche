#!/usr/bin/env bash

# First created: 
# Last modified: 2018 Jan 26

# Author: Feng Li
# email: fengggli@yahoo.com

PCI_ADDR="11:00.0"
NR_ELEMS=100000  # nr_elem for 4k
for VAL_LEN in 4096 65536 1048576 16777216 268435456
do
  for EXP_NAME in "put" "get" "get_direct"
  do

  echo "########### Runing ${EXP_NAME} with VAL_LEN=${VAL_LEN}, NR_ELEMS=${NR_ELEMS}, using pci_addr=${PCI_ADDR}"

  ./testing/kvstore/kvstore-perf --component nvmestore --pci_addr ${PCI_ADDR} --test=${EXP_NAME} --value_length=${VAL_LEN} --elements=${NR_ELEMS}

  done
  NR_ELEMS=$((NR_ELEMS/10))
done


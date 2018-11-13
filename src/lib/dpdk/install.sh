#!/bin/bash
echo "Installing to $1..."
mkdir -p $1 
cp -R ./dpdk/build/* $1


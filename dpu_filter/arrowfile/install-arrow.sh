#!/bin/bash
apt-get update

export DEBIAN_FRONTEND=noninteractive
#install tzdata package
apt-get install -y tzdata
# set your timezone
ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime
dpkg-reconfigure --frontend noninteractive tzdata

apt-get install -y git cmake build-essential

git config --global --add safe.directory /travis

# arrow

apt install -y -V ca-certificates lsb-release wget
wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
apt update
apt install -y -V libarrow-dev # For C++
apt install -y -V libarrow-glib-dev # For GLib (C)
apt install -y -V libarrow-dataset-dev # For Apache Arrow Dataset C++
apt install -y -V libarrow-dataset-glib-dev # For Apache Arrow Dataset GLib (C)
apt install -y -V libarrow-acero-dev # For Apache Arrow Acero
apt install -y -V libarrow-flight-dev # For Apache Arrow Flight C++
apt install -y -V libarrow-flight-glib-dev # For Apache Arrow Flight GLib (C)
apt install -y -V libgandiva-dev # For Gandiva C++
apt install -y -V libgandiva-glib-dev # For Gandiva GLib (C)
apt install -y -V libparquet-dev # For Apache Parquet C++
apt install -y -V libparquet-glib-dev # For Apache Parquet GLib (C)

apt install -y libtbb-dev # Intel thread building blocks

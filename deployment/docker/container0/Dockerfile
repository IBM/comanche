#
# Ubuntu Dockerfile
#
# https://github.com/dockerfile/ubuntu
#

# Pull base image.
FROM ubuntu:14.04

LABEL RUN docker run -i -t --privileged -v /dev/vfio/vfio:/dev/vfio/vfio -v /dev/vfio/5:/dev/vfio/5 -v /sys/bus/pci/drivers:/sys/bus/pci/drivers -v /sys/kernel/mm/hugepages:/sys/kernel/mm/hugepages -v /dev/hugepages:/dev/hugepages --name NAME -e NAME=NAME -e IMAGE=IMAGE IMAGE"

# Install.
RUN \
  sed -i 's/# \(.*multiverse$\)/\1/g' /etc/apt/sources.list && \
  apt-get update && \
  apt-get -y upgrade && \
  apt-get install -y build-essential && \
  apt-get install -y cmake gcc libpciaccess-dev make libcunit1-dev libaio-dev && \
  apt-get install -y libssl-dev libibverbs-dev librdmacm-dev libudev-dev uuid && \ 
  apt-get install -y htop && \
  apt-get install -y libibverbs-dev librdmacm-dev libnuma-dev && \
  apt-get install -y libunwind8 && \
  apt-get install -y software-properties-common && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /root
#ADD dpdk.tar.gz /root
#ADD spdk.tar.gz /root

COPY lib* /usr/lib/
COPY fstab /etc/fstab

RUN useradd -m batman && echo "batman:batman" | chpasswd && adduser batman sudo

USER batman
WORKDIR /home/batman

ADD comanche.tar.gz /home/batman
COPY startup.sh /home/batman/startup.sh
COPY setup.sh /home/batman/setup.sh

# Add files.
#ADD root/.bashrc /root/.bashrc
#ADD root/.gitconfig /root/.gitconfig
#ADD root/.scripts /root/.scripts

# Set environment variables.
ENV HOME /home/batman/

# Define working directory.
WORKDIR /home/batman

# Define default command.
CMD /bin/bash


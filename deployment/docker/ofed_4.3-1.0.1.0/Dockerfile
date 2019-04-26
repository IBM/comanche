# Use an official Ubuntu 16.04 as a parent image

FROM ubuntu:16.04
MAINTAINER travis.janssen@ibm.com 

# Set MOFED directory, image and working directory
ENV MOFED_DIR MLNX_OFED_LINUX-4.3-1.0.1.0-ubuntu16.04-x86_64
ENV MOFED_SITE_PLACE MLNX_OFED-4.3-1.0.1.0
ENV MOFED_IMAGE MLNX_OFED_LINUX-4.3-1.0.1.0-ubuntu16.04-x86_64.tgz

WORKDIR /tmp/

# Pick up some MOFED dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    net-tools \
    ethtool \
    perl \
    lsb-release \
    iproute2 \
    pciutils \
    libnl-route-3-200 \
    kmod \
    libnuma1 \
    lsof \
    linux-headers-4.4.0-92-generic \
    pkg-config flex gfortran chrpath graphviz automake tcl m4 dpatch libglib2.0-0 libgfortran3 debhelper autoconf swig bison autotools-dev tk libltdl-dev \
    python-libxml2 && \
    rm -rf /var/lib/apt/lists/*

# Download and install Mellanox OFED 4.2.1 for Ubuntu 16.04
RUN wget http://content.mellanox.com/ofed/${MOFED_SITE_PLACE}/${MOFED_IMAGE} && \
    tar -xzvf ${MOFED_IMAGE} && \
    ${MOFED_DIR}/mlnxofedinstall --user-space-only --without-fw-update --all -q && \
    cd .. && \
    rm -rf ${MOFED_DIR} && \
    rm -rf *.tgz


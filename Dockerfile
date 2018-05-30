# A dockerfile for minimal system requirement for auto-build comanche in travis

# Use an official Python runtime as a parent image
FROM ubuntu:16.04

# some dumb packages
RUN apt-get update \
  && apt-get -y install wget git

# enable sudo
RUN apt-get -y install sudo \
  && useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo


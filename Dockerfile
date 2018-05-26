# Template is from https://docs.docker.com/get-started/part2/ 
# you can also use WORKDIR, ADD, EXPOSE, ENV, CMD, see the link for more details
# Using dockerfile brings several benifits:
#   1. don't need do the time-consuming apt-get update in each travis-CI rebuild.
#   2, using different tags in docker, we can manage different ubuntu/centos environments for future comanche release
  

# Use an official Python runtime as a parent image
FROM ubuntu:16.04

# docker image will
RUN apt-get update \
  && apt-get install -y wget git 


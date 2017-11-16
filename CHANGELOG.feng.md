# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

Feng Li (lifen@iupui.edu)

## 2017-11-06
#### changed
- use CMAKE_SOURCE_DIR for out-of-tree build

#### added
- a wiki page on configuring qemu with nvme support with libvirt, [here](https://github.com/fengggli/comanche/wiki/vm)

#### note
- before running fetch-deps.sh
    export MAKEFLAGS="-j 8"
- uuid-dev is required by spdk build


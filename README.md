# Comanche
Framework for user-level compositional storage systems development. See [wiki](https://github.com/IBM/comanche/wiki) for more information.

[![Build Status](https://travis-ci.com/IBM/comanche.svg?branch=master)](https://travis-ci.com/IBM/comanche)
[![Build Status](https://travis-ci.com/IBM/comanche.svg?branch=unstable)](https://travis-ci.com/IBM/comanche)

NOTE: Comanche is in its early stages of development and while we welcome collaboration from the open source community, this effort is not for the faint hearted and requires a certain level of systems expertise.


HowTo
-----

* DPDK/SPDK

Comanche currently uses DPDK/SPDK.  You should install and test this
first (see [dpdk/usertools/dpdksetup.sh](https://github.com/spdk/dpdk/blob/spdk-17.05/usertools/dpdk-setup.sh)).  Note, it is possible to run
in a virtual machine (using UIO) but you will need to run your
programs as root.

* Prepare (one time - although it may change across checkouts)

```bash
( cd deps
  ./install-apts.sh
  ./fetch-deps.sh
)
```

* Build
**Now we enforce out-of-source build**

```bash
mkdir build
cd build
cmake ..
make

** To override the install destination

```bash
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/dist ..
```

## All libs will be installed in cmake installation prefix

```bash
make install
```

* Load modules (XMS) and attach NVMe devices to SPDK 
```bash
sudo ./load-module.sh
sudo ./tools/nvme_setup.sh /* optionally attach ALL Nvme devices to VFIO */
cmake .
make
```

* Build components in debug mode (e.g., with asan)

```bash
cmake -DCMAKE_BUILD_TYPE=Debug .
```

Other build options:

```bash
cmake -DCMAKE_BUILD_TYPE=ASAN .
cmake -DCMAKE_BUILD_TYPE=Release .
cmake -DCMAKE_BUILD_TYPE=NOOPT .
```

* Prepare to run (use tools scripts)

- SPDK/DPDK requires huge pages
  e.g. # echo 2000 > /proc/sys/vm/nr_hugepages
  

Tested Compilers and OS/HW
--------------------------

* Software 
  - Ubuntu 16.04.3 LTS (x86_64)
  - gcc 5.4
  - clang 3.8.0

* Hardware
  - Intel x86
  - Intel PC3700 and P4800X NVMe SSD
  - Samsung P1725a SSD


* Change compiler preference on Ubuntu with:
```
    'sudo update-alternatives --config c++'
    'sudo update-alternatives --config cc' 
```    
Compiling for debug
-------------------

DPDK
```bash
export EXTRA_CFLAGS='-O0 -g'
```
SPDK
```bash
CONFIG_DEBUG=y (command line or CONFIG file)
```


Conventions
-----------

C++ style guide - https://genode.org/documentation/developer-resources/coding_style

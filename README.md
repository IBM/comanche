# Comanche
Framework for user-level compositional storage systems development. See [wiki](https://github.com/IBM/comanche/wiki) for more information.

HowTo
-----

* Prepare (one time - although it may change across checkouts)

```bash
cd deps
./install-apts.sh
./fetch-deps.sh
```

* Build from root

```bash
source setenv.h
sudo ./tools/nvme_setup.sh /* optionally attach ALL Nvme devices to VFIO */
cmake .
make
```

* Build components in debug mode (e.g., with asan)

```bash
cmake -DCMAKE_BUILD_TYPE=Debug .
```

For mem leak checking - ASAN may hang block-nvme:

```bash
cmake -DCMAKE_BUILD_TYPE=ASAN .

cmake -DCMAKE_BUILD_TYPE=Release .
```

Tested Compilers and OS
------------------------

Ubuntu 16.04.3 LTS (x86_64)
gcc 5.4
clang 3.8.0

* Change compiler preference on Ubuntu with:
    'sudo update-alternatives --config c++'
    'sudo update-alternatives --config cc' 
    
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

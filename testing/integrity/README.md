# KVStore Test
## What is it?
KVStore Test is a set of test cases meant to ensure that different parts of KVStore functionality work correctly. These largely follow the pattern in the kvstore_intf.h APIs. 

## How to build it
This project uses CMake, so if you've set that up already, building for just this project can be done with `$ comanche/build/make test_kvstore`. 

## How to run it
If you've followed the README in Comanche's normal setup process, everything is run out of the build directory, including this test. The full command is `$ ./comanche/build/testing/integrity/test_kvstore`, which defaults to using the filestore component. If you want to use a different component, run with option `--component=<component_name>`. 

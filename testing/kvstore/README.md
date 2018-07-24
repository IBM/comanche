# KVStore Test
## What is it?
KVStore Test is a set of test cases meant to ensure that different parts of KVStore functionality work correctly. These largely follow the pattern in the kvstore_intf.h APIs. 

## How to build it
This project uses CMake, so if you've set that up already, building for just this project can be done with `$ comanche/build/make test_kvstore`. 

## How to run it
If you've followed the README in Comanche's normal setup process, everything is run out of the build directory, including this test. The full command is `$ ./comanche/build/testing/kvstore/test_kvstore`. Right now only a load test exists, but there's no failure criteria (obviously this will be updated).

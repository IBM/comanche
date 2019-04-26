#!/bin/bash
make install T=x86_64-native-linuxapp-gcc DESTDIR=./build EXTRA_CFLAGS="-g -fPIC"


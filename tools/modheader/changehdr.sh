#!/bin/bash
MODHEADER_BIN=$COMANCHE_HOME/tools/modheader/modheader

if [ ! $1 ] ; then
  echo "changehdr <new_header_file>"
fi

HEADERS=`find . -name "*.h"`
CXXFILES=`find . -name "*.c*"`
CFILES=`find . -name "*.c"`

for i in $HEADERS; do
    echo "Modifying header in file $i ..."
    $MODHEADER_BIN $i $1
done;
for i in $CXXFILES ; do
    echo "Modifying header in file $i ..."
    $MODHEADER_BIN $i $1
done;
for i in $CFILES; do
    echo "Modifying header in file $i ..."
    $MODHEADER_BIN $i $1
done;

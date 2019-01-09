#!/bin/bash
BIN_DIR=$HOME/comanche/build/dist/bin/

if [ -z "$1" ];
then
    SHARDS=16
else
    SHARDS=$1
fi

for i in $(seq 1 1 $SHARDS)
do
    PORT=$(( 11910 + $i ))
    CORE=%i
    echo "./dawn $i  --core 0 --backend mapstore --device mlx5_0 --port $PORT  --debug 0"
    $BIN_DIR/dawn $i  --core $i  --backend mapstore --device mlx5_0 --port $PORT  --debug 0 &
done

read foo

echo Closing down shards...
killall -9 dawn


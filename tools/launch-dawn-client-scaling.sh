#!/bin/bash
BIN_DIR=$HOME/comanche/build/dist/bin/
SERVER=10.0.0.51

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
    SHORT_CIRCUIT_BACKEND=1 $BIN_DIR/kvstore-perf  --component dawn --test throughput --server_address $SERVER:$PORT --debug_level 0 --device_name mlx5_0 --pool_name dax0 --path=/dev/ --elements 1000000 --cores 0-3 --nopin &> log.$i &
    pids[${i}]=$!
done

# wait for all pids
for pid in ${pids[*]}
do
    wait $pid
done

grep "IOPS" log.*
echo "All client complete!"


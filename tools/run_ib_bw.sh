#!/bin/bash

for i in {1..4}
do
    port=$((11910 + $i))
    ib_send_bw -x 0 -c RC -s 128 -D 10 -q 2 -p $port $0 &
done

pause
killall -9 ib_send_bw



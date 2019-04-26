#!/bin/bash
killall -9 ib_send_bw
for i in {1..16}
do
    port=$((18515 + $i))
    ib_send_bw -F -x 0 -c RC -s 128 -D 10 -p $port $1 $2 $3 $4 $5 $6&
done

read
killall -9 ib_send_bw



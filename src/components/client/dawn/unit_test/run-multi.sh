#!/bin/bash

for P in A B C ;
do
    gnome-terminal -x bash -c "./dawn-client-test1 10.0.0.21:11911 --pool "$P" && read" &
done

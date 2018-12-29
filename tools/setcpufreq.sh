#!/bin/bash
for c in {0..47}
do
    echo "cpufreq-set -g performance -c $c "
    cpufreq-set -g performance -c $c
    cpufreq-info -c $c
done
                   

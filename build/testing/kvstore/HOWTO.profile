LD_PRELOAD=/usr/lib/libprofiler.so PROFILESELECTED=1 CPUPROFILE=./cpu_profile ./kvstore-perf
google-pprof --gv kvstore-perf cpu_profile

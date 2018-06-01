LD_PRELOAD=/usr/lib/libprofiler.so PROFILESELECTED=1 PMEM_IS_PMEM_FORCE=1 ./kvstore-perf
google-pprof --gv kvstore-perf cpu.profile

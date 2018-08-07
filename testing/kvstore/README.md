# kvstore-perf
## What is it?
kvstore-perf is the performance testing project for Comanche. This is set up to be customizable in a variety of ways.

## How to: build
From the comanche/build directory, run:
`$ make kvstore-perf`

## How to: run
From the comanche/build directory, run:
`$ LD_PRELOAD=/usr/lib/libprofiler.so PROFILESELECTED=1 PMEM_IS_PMEM_FORCE=1 ./testing/kvstore/kvstore-perf`

Environment variables:
* LD_PRELOAD: load profiler library before anything else
* PROFILESELECTED: cpu profiler will only profile regions surrounded by ProfilerEnable/ProfilerDisable
* PMEM_IS_PMEM_FORCE: required for persistent memory tests; optional if not using persistent memory

### Run options
You can run with different command line options as input. Just add these to your run command with the format: `--<option_name>=<selection>`

* component: type of component you want to test (filestore, rocksdb, etc). Defaults to filestore since it has the least environmental dependencies.
* test: isolated test to run. Defaults to 'all'.
* cores: number of cores to use during test. Defaults to 1.
* time: time in seconds to run. This only effects certain types of tests. 
* path: location where pool should be created. 
* size: size of pool to create.
* flags: flags specified for pool creation.
* elements: number of data elements to create for testing; this value will also be passed to pool creation as the expected number of elements in the pool.

### Output
Information is stored in `results/<component_name>/results_<date>_<time>.json`
Example: `results/filestore/results_2018_08_06_14_28.json` would be the results file of an experiment conducted on the filestore component using the get_latency test on 8/6/2018 at 2:28pm.

# kvstore-perf overview
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

# Functionality
## Test sequence
Unless overwritten with the --test input, six tests will run, in the following order:
1) Put: tests IOPS of put operation
2) Get: tests IOPS of get operation
3) put_latency: tests put operation latency
4) get_latency: tests get operation latency
5) get_direct_latency: tests get_direct operation latency
6) put_direct_latency: tests put_direct operation latency

## Testing select operations 
If you're developing a component that doesn't support all the operations under tests, you can skip to the ones that are supported with the --test option. For instance, if only put_direct works, use --test="put_direct_latency" and all other tests will be skipped apart from that one.

## Options
You can run with different command line options as input. Just add these to your run command with the format: `--<option_name>=<selection>`

* component: type of component you want to test (filestore, rocksdb, etc). Defaults to filestore since it has the least environmental dependencies.
* test: isolated test to run. Defaults to 'all'.
* cores: comma separated ranges of indexes of cores to use during test. Defaults to 0. A range may be specified by a single index, and pair of indexes separated by a hyphen, or an index followed by a colon and a count of additional indexes. These examples all specify nodes 2 through 4 inclusive: "2,3,4", "2-4", "2:3".
* devices: comma-separated ranges of devices to use during test. Defaults to the value of core. Each identifier is a dotted pair of numa zone and index, e.g. "1.2". For comaptibility with cores, a simple index number is accepted and implies numa node 0. These examples all specify device indexes 2 through 4 inclusive in numa node 0: "2,3,4", "0.2:3". These examples all specify devices 2 thourgh 4 inclusive on numa node 1: "1.2,1.3,1.4", "1.2-1.4", "1.2:3".  When using hstore, the actual dax device names are concatenations of the device_name option with <node>.<index> values specified by this option. In the node 0 example above, with device_name /dev/dax, the device paths are /dev/dax0.2 through /dev/dax0.4 inclusive.
* time: time in seconds to run. This only effects certain types of tests. 
* path: location where pool should be created. 
* size: size of pool to create. This is in bytes, so it may be a large number.
* flags: flags specified for pool creation.
* elements: number of data elements to create for testing; this value will also be passed to pool creation as the expected number of elements in the pool.
* key_length: length of the key in the Key-Value Store to test (defaults to length 8)
* value_length: length of the value in the Key-Value Store (defaults to length 64)
* bins: certain tests (the ones with _latency in them in particular) compute statistics at runtime. Use this argument to specify the number of bins. Defaults to 100.
* latency_range_min: latency bin minimum threshold for statistics. Defaults to 1ns.
* latency_range_max: latency bin maximum threshold for statistics. Defaults to 10ms.
* debug_level: optional debug parameter that some components use. Defaults to 0.
* owner: string for component registration. Defaults to "owner".
* server_address: for client components only - server address and port of component server, as one string. No default value.
* device_name: some components can access multiple hardware instances on the same machine. Specify the device name here, like "mlx5_0". No default value.

## Output
Information is stored in `results/<component_name>/results_<date>_<time>.json`
Example: `results/filestore/results_2018_08_06_14_28.json` would be the results file of an experiment conducted on the filestore component using the get_latency test on 8/6/2018 at 2:28pm.

# Comanche Docker image
This is a containerized version of the Comanche project, dependencies and all. This requires a previous build of the OFED4.3-1.0.1.0 docker image as a requirement. Additional setup and configuration instructions for Docker are there.

# How to build it
`sudo docker build --network=host --tag="comanche" .`

If you need to update the core code from Comanche, run with the `--no-cache` option. **Warning**: this will rebuild the entire image from scratch.

# How to run it
`sudo docker run -it --privileged --network=host comanche`

## Example run: unit tests
From container:
```bash
$ cd testing/integrity
$ ./test_kvstore
```

## Example run: performance tests
From container:
```bash
$ cd testing/kvstore
$ ./kvstore_perf
```

# Comanche Docker image
This is a containerized version of the Comanche project, dependencies and all. This requires a previous build of the OFED4.3-1.0.1.0 docker image as a requirement. Additional setup and configuration instructions for Docker are there.

# How to build it
`sudo docker build --network=host --tag="comanche" .`

# How to run it
`sudo docker run -it --privileged --network=host comanche`

## Example run: unit tests
```bash
$ cd testing/integrity
$ ./test_kvstore
```

## Example run: performance tests
```bash
$ cd testing/kvstore
$ ./kvstore_perf
```

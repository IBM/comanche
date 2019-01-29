# OFED-4.3-1.0.1.0 Docker container 
This exists to check RDMA functionality inside a container.

# How to Build the image
## Requirements: host system
1. Docker installed. Originally created and tested with version 18.06.1-ce, build e68fc7a
2. DNS information is known (`$ cat /etc/resolv.conf`)
3. IP forwarding enabled in kernel (`$ sysctl net.ipv4.conf.all.forwarding=1`)
4. Docker allowed to forward traffic from outside the host machine (`$ sudo iptables -P FORWARD ACCEPT`)

## Setup: docker0 bridge
1. Make sure docker0 is visible in ifconfig output
2. Add/edit file: /etc/docker/daemon.json based on info below, changing values as appropriate:

```json
"bip": "10.0.0.55/24",
"mtu": 9000,
"dns": ["9.1.44.254", "9.1.32.254"]
```

Explanation:
* bip = IP address for docker0 bridge
* mtu = maximum packet length
* dns = dns servers to use

3. Save and restart Docker (`$ sudo /etc/init.d/docker restart`)


# Build the image
Try this first: `$ sudo docker build -tag="ofed4.3-1.0.1.0"`

If you see DNS errors like "Temporary failure resolving 'deb.debian.org'", run with `-network=host' input option: `$ sudo docker build -network=host -tag="ofed4.3-1.0.1.0`

This will build an image called ofed4.3-1.0.1.0. Verify it exists with `$ sudo docker images`

# How to run Docker image
`$ sudo docker run -it --privileged -network=host ofed4.3-1.0.1.0 sh`

Options explanation:
* it: interactive TTY. Otherwise, it'll just run and stop.
* privileged: elevate container privileges. Otherwise, can't get context for RDMA device.
* network=host: allow docker to use Docker host network stack. Otherwise, container can't open file descriptor for socket connection. Note that this may be able to be removed on some networks.
* sh: launch command interpreter (shell) at container start

## Exiting container
`exit` from command line

## Sample commands
`$ ib_send_bw 10.0.0.61` run bandwidth test from container to server at 10.0.0.61

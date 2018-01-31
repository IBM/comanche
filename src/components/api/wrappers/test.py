#!/usr/bin/python -i
import Block
import rlcompleter, readline
readline.parse_and_bind("tab: complete")

device = Block.Block("86:00.0",2,"libcomanche-blknvme.so")

buffer = device.allocate_io_buffer(4096,32,-1)

info = device.get_volume_info()

info

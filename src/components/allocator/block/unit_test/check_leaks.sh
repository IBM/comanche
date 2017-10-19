#!/bin/bash
MALLOC_TRACE=trace.raw.log ./block-allocator-test1
mtrace ./block-allocator-test1 ./trace.raw.log

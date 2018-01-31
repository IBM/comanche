Author: Daniel G. Waddington (daniel.waddington@ibm.com)
Description: Component for Blob store, large values > 1GB
Notes:

This blob store is designed for the following criteria.

* Large blobs (no support for meta-data, value block sharing)
* 4K underlying block size
* Zero-copy
* Opening cursor allocates memory for total blob value size



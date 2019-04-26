This demo shows how to move data from the DAWN RDMA-attached store
directly into the GPU.

To run this demo, you will need to have the nv_peer_mem kernel module
installed and loaded.

You must also make sure you have NOT enabled the IOMMU (check /etc/default/grub).

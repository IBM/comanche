
#!/bin/bash
boot_path=./kernels/
sudo qemu-system-x86_64 -m 16G \
                   -cpu Broadwell \
                   -kernel ${boot_path}/vmlinuz-4.9.33-040933-generic \
                   -initrd ${boot_path}/initrd.img-4.9.33-040933-generic \
                   -machine q35,iommu=on\
                   -nographic -serial mon:stdio \
                   -hda ./ubuntu1604.qcow2 \
                   -drive file=./nvme.img,if=none,id=drv0 \
                   -device nvme,drive=drv0,serial=foo \
                   -smp 32 \
                   -append 'root=/dev/sda1 console=ttyS0 default_hugepagesz=1GB hugepagesz=1G hugepages=1 intel_iommu=on iommu=pt vfio_iommu_type1.allow_unsafe_interrupts=1' \
                   -net user,vlan=0 -net nic,vlan=0 \
                   -fsdev local,id=mdev,path=/home/fengggli,security_model=none -device virtio-9p-pci,fsdev=mdev,mount_tag=host_mount \
                   -enable-kvm \
                   -vga std \
                   -redir tcp:2222::22
                #-append 'root=/dev/sda1 console=ttyS0 default_hugepagesz=1G hugepagesz=1G hugepages=4 hugepagesz=2M hugepages=512 intel_iommu=on iommu=pt vfio_iommu_type1.allow_unsafe_interrupts=1' \



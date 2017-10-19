KERNEL_VER = `uname -r`
BUILD = `date +%Y%m%d.%k%m`

TARGET = copager

ccflags-y := -I$(HOME)/comanche/src/components/collaborative-pager/src -g -Wall 

copager-objs := copager_dev.o copager_mmap.o 

obj-m += ${TARGET}.o

all: ${TARGET}.ko 

KDIR=/lib/modules/$(shell uname -r)/build/


copager.ko: copager_dev.c copager_mmap.c
	make -C $(KDIR) M=$(PWD) modules

clean:
	make -C $(KDIR) M=$(PWD) clean;


install:
	sudo insmod copager.ko

uninstall:
	sudo rmmod copager


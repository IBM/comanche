
KERNEL_VER = `uname -r`
BUILD = `date +%Y%m%d.%k%m`

ccflags-y := -g -Wall

obj-m += xmsmod.o

xmsmod-objs := xms.o page_walk.o

all: xms.ko

xms.ko:
	make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) modules

clean:
	make -C /lib/modules/$(shell uname -r)/build/ M=$(PWD) clean

client:
	g++ -g -std=c++11 -DCONFIG_DEBUG -O2 -o testcli ./testcli.cc -I$(COMANCHE_HOME)/src/lib/common/include/


.PHONY: client

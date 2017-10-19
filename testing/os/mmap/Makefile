all: mmap_test

mmap_test: main.cc
	g++ -std=c++11 -g -I../../../include -DCONFIG_DEBUG -o mmap_test main.cc -lpthread

clean:
	rm -f mmap_test



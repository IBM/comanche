all: mprotect_test

mprotect_test: main.cc
	g++ -O2 -std=c++11 -I../../../include -DCONFIG_DEBUG -o mprotect_test main.cc -lpthread

clean:
	rm -f mprotect_test



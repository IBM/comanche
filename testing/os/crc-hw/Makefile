all: crc_test

crc_test: main.cc
	g++ -O3 -msse4.2 -DCONFIG_DEBUG -I../../../src/lib/common/include/ -o crc_test main.cc -lpthread -lisal

clean:
	rm -f crc_test



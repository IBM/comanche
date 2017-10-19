all: sha_test

sha_test: sha.cc
	g++ -DCONFIG_DEBUG -I../../../include -o sha_test sha.cc -lpthread

clean:
	rm -f sha_test



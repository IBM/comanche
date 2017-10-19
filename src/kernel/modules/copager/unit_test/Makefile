CXXFLAGS += -DCONFIG_DEBUG -I../../../../lib/common/include/

all: copager-test


copager-test:
	g++ -std=c++14 $(CXXFLAGS) -o copager-test test1.cc -lpthread

clean:
	rm -f copager-test

.PHONY: copager-test

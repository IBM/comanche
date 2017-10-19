CXXFLAGS += -I/usr/lib/jvm/java-8-openjdk-amd64/include -I/usr/lib/jvm/java-8-openjdk-amd64/include/linux -Wl,--no-undefined
CXXFLAGS += -D_REENTRANT

JAVA_LIBS = -Wl,-rpath=/usr/lib/jvm/java-8-openjdk-amd64/lib/amd64 -L/usr/lib/jvm/java-8-openjdk-amd64/lib/amd64 -Wl,-rpath=/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server/ -L/usr/lib/jvm/java-8-openjdk-amd64/jre/lib/amd64/server/ -ljawt -ljvm

all: jnitest

jnitest:
	g++ -O2 -std=c++14 -g  $(CXXFLAGS) -o jnitest main.cc $(JAVA_LIBS)

clean:
	rm -Rf jnitest


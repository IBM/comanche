INCLUDES = -I$(HOME)/kivati/src/lib/common/include

all: uf2

uf1:
	gcc -o uf1 userfaultfd.c -lpthread

uf2:
	gcc $(INCLUDES) -o uf2 uf2.c -lpthread

clean:
	rm -f uf2 uf1 

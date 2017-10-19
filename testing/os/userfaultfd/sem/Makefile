include ../../../mk/global.mk

all: sem_test

sem_test: 
	g++ main.cc $(KIVATI_INCLUDES) -o sem_test -lpthread -L$(KIVATI_HOME)/lib -lcommon

clean:
	rm -f *.o sem_test

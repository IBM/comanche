COMANCHE_HOME=$(HOME)/comanche

COMANCHE_INCLUDES += -I$(COMANCHE_HOME)/src/lib/common/include
COMANCHE_INCLUDES += -I$(COMANCHE_HOME)/src/lib/core/include -I$(COMANCHE_HOME)/src/lib/component/include

COMANCHE_LIBS 		= -L$(COMANCHE_HOME)/lib -L/usr/local/lib -lnanomsg -lnuma -ldl -lrt

COMANCHE_EXT_LIBS = -L/usr/local/lib -lnanomsg -lnuma -ldl -lrt -lpthread -lczmq -lzmq  -lzyre -lrdmacm -libverbs 
#DEBUG_FLAGS = -DCONFIG_BUILD_DEBUG

CXXFLAGS += -g -std=c++14 $(DEBUG_FLAGS) $(COMANCHE_INCLUDES)
CFLAGS += -g $(DEBUG_FLAGS) $(COMANCHE_INCLUDES) 

LIBS += $(COMANCHE_LIBS)


EXE=./run
MULTI=./multi
OPT=./opt
FEAT=./feat
MULTIOPT=./multiopt
MAINDIR=.

SRCDIR=.
OBJDIR=.

SOURCES=$(wildcard $(SRCDIR)/main.cc)
SOURCES_MULTI=$(wildcard $(SRCDIR)/multi.cpp)
SOURCES_OPT=$(wildcard $(SRCDIR)/opt.cc)
SOURCES_FEAT=$(wildcard $(SRCDIR)/feat.cc)
SOURCES_MULTIOPT=$(wildcard $(SRCDIR)/multiopt.cpp)
OBJECTS=$(SOURCES:%.cc=%.o)
OBJECTS_MULTI=$(SOURCES_MULTI:%.cpp=%.o)
OBJECTS_OPT=$(SOURCES_OPT:%.cc=%.o)
OBJECTS_FEAT=$(SOURCES_FEAT:%.cc=%.o)
OBJECTS_MULTIOPT=$(SOURCES_MULTIOPT:%.cpp=%.o)
HEADERS=$(SOURCES:.cc=.h) $(SOURCES_MULTI:.cpp=.h) $(SOURCES_OPT:.cc=.h) $(SOURCES_FEAT:.cc=.h) $(SOURCES_MULTIOPT:.cpp=.h)

CC          = g++
CXX         = g++
CXXFLAGS    = -O3 -Wall -fPIC -Wshadow -Woverloaded-virtual
LD          = g++
LDFLAGS     = -O3
SOFLAGS     = -shared
CXXFLAGS    = -O3 -Wall -fPIC -Wshadow -Woverloaded-virtual
LDFLAGS     = -O3
ROOTLIBS    = $(shell root-config --libs)
ROOTCFLAGS  = $(shell root-config --cflags)
CXXFLAGS   += $(ROOTCFLAGS)
CFLAGS      = $(ROOTCFLAGS) -Wall -Wno-unused-function -ltbb -O3  -fPIC -fno-var-tracking
CFLAGS     +=
EXTRAFLAGS  = -fPIC -ITMultiDrawTreePlayer -Wunused-variable -lTMVA -lEG -lGenVector -lXMLIO -lMLP -lTreePlayer
EXTRAFLAGS +=

all: $(EXE) $(MULTI) $(OPT) $(FEAT) $(MULTIOPT)

$(EXE): $(OBJECTS)
	$(LD) $(CXXFLAGS) $(LDFLAGS) $(OBJECTS) $(ROOTLIBS) $(EXTRAFLAGS) -o $@

$(MULTI): $(OBJECTS_MULTI)
	$(LD) $(CXXFLAGS) $(LDFLAGS) $(OBJECTS_MULTI) $(ROOTLIBS) $(EXTRAFLAGS) -o $@

$(OPT): $(OBJECTS_OPT)
	$(LD) $(CXXFLAGS) $(LDFLAGS) $(OBJECTS_OPT) $(ROOTLIBS) $(EXTRAFLAGS) -o $@

$(FEAT): $(OBJECTS_FEAT)
	$(LD) $(CXXFLAGS) $(LDFLAGS) $(OBJECTS_FEAT) $(ROOTLIBS) $(EXTRAFLAGS) -o $@

$(MULTIOPT): $(OBJECTS_MULTIOPT)
	$(LD) $(CXXFLAGS) $(LDFLAGS) $(OBJECTS_MULTIOPT) $(ROOTLIBS) $(EXTRAFLAGS) -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cc
	$(CC) $(CFLAGS) $< -c -o $@

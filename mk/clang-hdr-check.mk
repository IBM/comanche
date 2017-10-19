# filter removes warning around excessive padding
HEADER_FILES=$(shell find . -type f -name '*.h')
SYSTEM_INCLUDES=-I/usr/include/c++/5 -I/usr/include/x86_64-linux-gnu/c++/5/ -I/usr/include/linux -I/usr/include/c++/5/tr1

SKEL:=$(foreach hdr,$(HEADER_FILES),"\#include \"$(hdr)\"\\n")

tidy:
	@echo $(SKEL) > .tmp.cc
	clang-tidy --header-filter=./ -checks='+*,-clang-analyzer-optin.performance.Padding' .tmp.cc -- -std=c++11 -I$(COMANCHE_HOME)/include -I$(COMANCHE_HOME)/src/include -I$(COMANCHE_HOME)/dune/libdune -I$(COMANCHE_HOME)/EASTL/include -I$(COMANCHE_HOME)/EASTL/test/packages/EABase/include/Common $(SYSTEM_INCLUDES) 

# this will use COMANCHE_HOME/.clang-format file
# see https://clang.llvm.org/docs/ClangFormatStyleOptions.html for format file info
#
format:
	clang-format -i $(HEADER_FILES)

.PHONY: tidy format


#include "ustack_client.h"
#include <common/logging.h>
static Ustack_client *_this_client;

/** Constructor*/
void __attribute__((constructor)) ustack_ctor(); 

/** Destructor*/
void __attribute__((destructor)) ustack_dtor(); 

void ustack_ctor(){
  PINF("ustack preload");
  _this_client = new Ustack_client("ipc:///tmp//kv-ustack.ipc", 64);
}
void ustack_dtor(){
  PINF("ustack preload --unloading");
  delete _this_client;
}


/* Open will also keep the fuse-fd*/
int ustack_open(const char * pathname, int flags, mode_t mode);

int ustack_open(const char * pathname, mode_t mode);

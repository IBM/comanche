#include "ustack_client.h"
#include <common/logging.h>
#include <stdarg.h>
static Ustack_client *_this_client = NULL;

typedef int (*close_t)(int);
typedef int (*open64_t)(const char *pathname, int flags, ...);

open64_t orig_open64;
close_t orig_close;

/** Constructor*/
void __attribute__((constructor)) ustack_ctor(); 

/** Destructor*/
void __attribute__((destructor)) ustack_dtor(); 

void ustack_ctor(){
  orig_close = (close_t)(dlsym(RTLD_NEXT, "close"));
  assert(orig_close);

  orig_open64 = (open64_t)(dlsym(RTLD_NEXT, "open64"));
  assert(orig_open64);

  PINF("Original open64/close intialized");
  _this_client = new Ustack_client("ipc:///tmp//kv-ustack.ipc", 64);
  PINF("Ustack Preloaded");
}

void ustack_dtor(){
  PINF("ustack preload --unloading");
  delete _this_client;
}


int open64 (const char * pathname, int flags, ...){
// int open(const char *pathname, int flags, mode_t mode){
 PLOG("Ustack open64 intercepted at path (%s)", pathname);

 // full path to fd
 int fd = -1;
 va_list vl;
 va_start(vl, flags);
 
 // fall into the mountdir?
 fd = orig_open64(pathname, flags, vl);
 
 //if( _is_ustack_path(pathname, fullpath) && fd >0){
 uint64_t fuse_fh = 0;
 if(0 == ioctl(fd, USTACK_GET_FUSE_FH, &fuse_fh)){
   PLOG("{ustack_client]: register file %s with fd %d, fuse_fh = %lu", pathname, fd, fuse_fh);
   assert(fuse_fh > 0);
   _fd_map.insert(std::pair<int, uint64_t>(fd, fuse_fh));
 }

 va_end(vl);
 return fd;
}

int close(int fd){
  int ret = -1;

  ret =  orig_close(fd);

  PLOG("Ustack close intercepted for fd (%d)", fd);
#if 0 // I couln't use stl here, TODO: use array
  if(ret == 0){
  try{
    PLOG("0");
    if(_fd_map.find(fd) != _fd_map.end()){
      PLOG("1");
      _fd_map.erase(fd);
      PLOG("2");
    }
  }
  catch(...){
    PWRN("ustack_client]remove fd failed, it's not a managed fuse-fd?");
  }
  }
#endif
  return ret;
};

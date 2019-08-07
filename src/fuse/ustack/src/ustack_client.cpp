#include <common/logging.h>
#include <stdarg.h>
#include "ustack_client.h"
static constexpr int k_nr_files   = 64; /** Intial capacity(num of files)*/
static Ustack_client *  _this_client = NULL;

static int *_fd_array = NULL;

enum {
  FD_ARRAY_INVALID = -1,
  FD_ARRAY_OK      = 0,
};
static int _fd_array_initialized = FD_ARRAY_INVALID;

enum {
  FUSE_FD_INVALID = -1,
};

typedef int (*close_t)(int);
typedef int (*open64_t)(const char *pathname, int flags, ...);
typedef void *(*malloc_t)(size_t size);
typedef void (*free_t)(void *ptr);
/*typedef ssize_t (*read_t)(int fd, void *buf, size_t count);*/
/*typedef ssize_t (*write_t)(int fd, const void *buf, size_t count);*/
typedef ssize_t (*pread_t)(int fd, void *buf, size_t count, off_t offset);
typedef ssize_t (*pwrite_t)(int fd, const void *buf, size_t count, off_t offset);
typedef void *(*mmap_t)(void *addr, size_t length, int prot, int flags, int fd, off_t offset);
typedef int (*munmap_t)(void *addr, size_t length);


open64_t orig_open64;
close_t  orig_close;
malloc_t orig_malloc;
free_t   orig_free;
/*read_t   orig_read;*/
/*write_t  orig_write;*/
pread_t   orig_pread;
pwrite_t  orig_pwrite;
mmap_t orig_mmap;
munmap_t orig_munmap;

/** Constructor*/
void __attribute__((constructor)) ustack_ctor();

/** Destructor*/
void __attribute__((destructor)) ustack_dtor();

void ustack_ctor()
{
  size_t fd_array_size;

  /* Save default calls*/
  orig_close = (close_t)(dlsym(RTLD_NEXT, "close"));
  assert(orig_close);

  orig_open64 = (open64_t)(dlsym(RTLD_NEXT, "open64"));
  assert(orig_open64);

  orig_malloc = (malloc_t)(dlsym(RTLD_NEXT, "malloc"));
  assert(orig_malloc);

  orig_free = (free_t)(dlsym(RTLD_NEXT, "free"));
  assert(orig_free);

/*  orig_read = (read_t)(dlsym(RTLD_NEXT, "read"));*/
  //assert(orig_read);

  //orig_write = (write_t)(dlsym(RTLD_NEXT, "write"));
  /*assert(orig_write);*/

  orig_pread = (pread_t)(dlsym(RTLD_NEXT, "pread"));
  assert(orig_pread);

  orig_pwrite = (pwrite_t)(dlsym(RTLD_NEXT, "pwrite"));
  assert(orig_pwrite);

  orig_mmap = (mmap_t)(dlsym(RTLD_NEXT, "mmap"));
  assert(orig_mmap);

  orig_munmap = (munmap_t)(dlsym(RTLD_NEXT, "munmap"));
  assert(orig_munmap);

  PINF("Original open64/close/malloc intialized");

  /* Initialize ustack
   * Attention, open/close is also used for xms
   **/
  _this_client = new Ustack_client("ipc:///tmp//kv-ustack.ipc", 8192); // 32M IO memory
  PINF("Ustack Preloaded");

  /* Allocate space for fd mappings*/
  fd_array_size = k_nr_files * sizeof(int);
  _fd_array     = (int *) orig_malloc(fd_array_size);
  assert(_fd_array);
  memset((void *) (_fd_array), 0xff, fd_array_size);

  _fd_array_initialized = FD_ARRAY_OK;
  PINF("fd_array init at %p", _fd_array);

  assert(_this_client->get_uipc_channel());
}

void ustack_dtor()
{
  _fd_array_initialized = FD_ARRAY_INVALID;
  if (_fd_array) orig_free(_fd_array);
  PINF("ustack preload --unloading");
  delete _this_client;
}



/**
 * Overwriting of open.
 *
 * If the file falls into fuse-enabled folder, tracks the fd
 */
int open64(const char *pathname, int flags, ...)
{
  // int open(const char *pathname, int flags, mode_t mode){

  // full path to fd
  int      fd = -1;
  uint64_t fuse_fh;

  /* Get the mode if it's a creation*/
  mode_t mode;
  va_list  vl;
  if(flags & O_CREAT){
    va_start(vl, flags);
    mode = (mode_t)va_arg(vl, mode_t);
    va_end(vl);

    fd = orig_open64(pathname, flags, mode);
  }
  else{
    fd = orig_open64(pathname, flags);
  }


  // fall into the mountdir?
  if (fd >= k_nr_files) {
    PERR("needs to increase k_nr_files(currently =%d), limit(%d)", fd, k_nr_files);
    goto end;
  }

  PLOG("Ustack open64 intercepted at path (%s), fd(%d)", pathname, fd);

  fuse_fh = 0;
  if (0 == ioctl(fd, USTACK_GET_FUSE_FH, &fuse_fh)) {
    assert(_fd_array_initialized == FD_ARRAY_OK);
    PLOG("{ustack_client]: register file %s with fd %d, fuse_fh = %lu",
         pathname, fd, fuse_fh);
    PINF("fd_array is at %p", _fd_array);
    assert(fuse_fh > 0);
    _fd_array[fd] = fuse_fh;
  }

end:

  return fd;
}

#if 0
void * malloc(size_t n_bytes) {
  void * ret = NULL;
  if(_fd_array_initialized == FD_ARRAY_OK){
    ret =  _this_client->malloc(n_bytes);
    PLOG("malloc %lu bytes using ustack memory at (%p)", n_bytes, ret);
  }
  else{ 
    ret =  orig_malloc(n_bytes);
    PLOG("malloc %lu bytes using regular heap memory at (%p)", n_bytes, ret);
  }
  return ret;
}

void free(void * ptr) {
  status_t ret = E_FAIL;
  if(_fd_array_initialized == FD_ARRAY_OK && S_OK == _this_client->free(ptr)){

    PLOG("free ustack memory at (%p)", ptr);
      return;
  }
  else{
    PLOG("free regular heap memory at (%p)", ptr);
    orig_free(ptr);
    return;
  }
}
#endif

/** Overwriting of posix close*/
int close(int fd)
{
  int ret = -1;

  ret = orig_close(fd);

  PLOG("Ustack close intercepted for fd (%d)", fd);

  if (ret == 0 && fd >= 0 && _fd_array_initialized == FD_ARRAY_OK) {
    _fd_array[fd] = FUSE_FD_INVALID;
  }
  return ret;
};




void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset){
  void * ret;
  if(flags & MAP_HUGETLB){
    ret =  _this_client->malloc(length);
    assert(memset(ret, 0, length));
    PLOG("intercep HUGETLB mmap with flags(%x) size(%ld), return %p", flags, length, ret);
  }
  else{
    PLOG("[mmap]: fall back mmap with flags(%x) size(%ld)", flags, length);
    ret =  orig_mmap(addr, length, prot, flags, fd, offset);
  }
  return ret;
}

int munmap(void *addr, size_t length){
  int ret = -1;

  PLOG("[munmap with addr %p, length = %ld", addr, length);
  ret =  orig_munmap(addr, length);
  if(ret){ // is ustack-managed
    _this_client->free(addr);
    ret = 0;
  }

  return ret;
}

/**
 * File write and read
 * TODO: intercept posix ops
 *
 */
ssize_t pwrite(int fd, const void *buf, size_t count, off_t file_off)
{
  int search;
  if (_fd_array_initialized == FD_ARRAY_OK &&
      (search = _fd_array[fd]) != FUSE_FD_INVALID) {
    uint64_t fuse_fh = search;
    PLOG("[stack-write]: try to write from %p to fuse_fh %lu, size %lu, offset %lu", buf,
         fuse_fh, count, file_off);
    return _this_client->write(fuse_fh, buf, count, file_off);
  }
  else {
    /* regular file */
    PLOG("[stack-write]: fall back to orig_pwrite fd(%d)", fd);
    return orig_pwrite(fd, buf, count, file_off);
  }
}

ssize_t pread(int fd, void *buf, size_t count, off_t file_off)
{
  int search;
  if (_fd_array_initialized == FD_ARRAY_OK &&
      (search = _fd_array[fd]) != FUSE_FD_INVALID) {
    uint64_t fuse_fh = search;

    PLOG("[stack-read]: try to write from %p to fuse_fh %lu, size %lu, offset %lu", buf,
         fuse_fh, count, file_off);
    return _this_client->read(fuse_fh, buf, count, file_off);
  }
  else {
    /* regular file */
    PLOG("[stack-read]: fall back to orig_read fd(%d), buf(%p), count(%ld)", fd, buf, count);
    return orig_pread(fd, buf, count, file_off);
  }
}

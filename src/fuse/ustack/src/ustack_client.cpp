#include <common/logging.h>
#include <stdarg.h>
#include "ustack_client.h"
static constexpr size_t k_nr_files   = 64; /** Intial capacity(num of files)*/
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
typedef ssize_t (*read_t)(int fd, void *buf, size_t count);
typedef ssize_t (*write_t)(int fd, const void *buf, size_t count);

open64_t orig_open64;
close_t  orig_close;
malloc_t orig_malloc;
free_t   orig_free;
read_t   orig_read;
write_t  orig_write;

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

  orig_read = (read_t)(dlsym(RTLD_NEXT, "read"));
  assert(orig_read);

  orig_write = (write_t)(dlsym(RTLD_NEXT, "write"));
  assert(orig_write);

  PINF("Original open64/close/malloc intialized");

  /* Initialize ustack
   * Attention, open/close is also used for xms
   **/
  _this_client = new Ustack_client("ipc:///tmp//kv-ustack.ipc", 64);
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
  PLOG("Ustack open64 intercepted at path (%s)", pathname);

  // full path to fd
  int      fd = -1;
  uint64_t fuse_fh;
  va_list  vl;
  va_start(vl, flags);

  // fall into the mountdir?
  fd = orig_open64(pathname, flags, vl);
  if (fd >= k_nr_files) {
    PERR("needs to increase k_nr_files(currently =%lu)", k_nr_files);
    goto end;
  }

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
  va_end(vl);
  return fd;
}

#if 0
void * malloc(size_t n_bytes) {
  if(_fd_array_initialized == FD_ARRAY_OK){
    return _this_client->malloc(n_bytes);
  }
  else return orig_malloc(n_bytes);
}

void free(void * ptr) {
  status_t ret = E_FAIL;
  if(_fd_array_initialized == FD_ARRAY_OK && S_OK == _this_client->free(ptr))
      return;
  else{
    PWRN("ustack_client: intercep free not handled %p", ptr);
    orig_free(ptr);
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

/**
 * File write and read
 * TODO: intercept posix ops
 *
 * the message will be like(fd, phys(buf), count)
 */
ssize_t write(int fd, const void *buf, size_t count)
{
  int search;
  if (_fd_array_initialized == FD_ARRAY_OK &&
      (search = _fd_array[fd]) != FUSE_FD_INVALID) {
    uint64_t fuse_fh = search;
    PLOG("[stack-write]: try to write from %p to fuse_fh %lu, size %lu", buf,
         fuse_fh, count);
    return _this_client->write(fuse_fh, buf, count);
  }
  else {
    /* regular file */
    PLOG("[stack-write]: fall back to orig_write fd(%d)", fd);
    return orig_write(fd, buf, count);
  }
}

ssize_t read(int fd, void *buf, size_t count)
{
  int search;
  if (_fd_array_initialized == FD_ARRAY_OK &&
      (search = _fd_array[fd]) != FUSE_FD_INVALID) {
    uint64_t fuse_fh = search;

    PLOG("[stack-write]: try to write from %p to fuse_fh %lu, size %lu", buf,
         fuse_fh, count);
    return _this_client->read(fuse_fh, buf, count);
  }
  else {
    /* regular file */
    PLOG("[stack-write]: fall back to orig_write fd(%d)", fd);
    return orig_read(fd, buf, count);
  }
}

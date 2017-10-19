extern "C"
{
#include "fs.h"
}

#include <api/block_itf.h>
#include <common/logging.h>
#include <string>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>

#define DEFAULT_NR_BUFS 1024

extern Component::IBlock_device * g_block_layer;

extern "C"
{
  
struct glue_context {
  int block_size;
  int rdwt_err;
  struct buf buf_head;
};

static struct glue_context *ctx = NULL;

#define BUF_LIST_FOREACH(bp) for ((bp) = ctx->buf_head.lmfs_next; (bp) != &ctx->buf_head; (bp) = (bp)->lmfs_next)

static void buf_list_init(void) {
  ctx->buf_head.lmfs_next = &ctx->buf_head;
  ctx->buf_head.lmfs_prev = &ctx->buf_head;
}

static int buf_list_empty(void) {
  return ctx->buf_head.lmfs_next == &ctx->buf_head;
}

static void buf_list_add(struct buf *bp) {
  bp->lmfs_prev = &ctx->buf_head;
  bp->lmfs_next = ctx->buf_head.lmfs_next;
  ctx->buf_head.lmfs_next = bp;
  bp->lmfs_next->lmfs_prev = bp;
}

static void buf_list_remove(struct buf *bp) {
  bp->lmfs_prev->lmfs_next = bp->lmfs_next;
  bp->lmfs_next->lmfs_prev = bp->lmfs_prev;
  bp->lmfs_next = NULL;
  bp->lmfs_prev = NULL;
}

void panic(const char *format, ...) {
  va_list ap;

  va_start(ap, format);
  vfprintf(stderr, format, ap);
  va_end(ap);
  fprintf(stderr, "\n");

  exit(EXIT_FAILURE);
}

void util_stacktrace(void)
{
}

time_t clock_time(struct timespec *tv) {
  time_t t = time(NULL);
  if (tv) {
    tv->tv_sec  = t;
    tv->tv_nsec = 0;
  }
  return t;
}

void glue_setup() {
  
  ctx = (struct glue_context*) ::malloc(sizeof(struct glue_context));
  if (ctx == NULL) {
    panic("couldn't alloc memory");
  }

  assert(g_block_layer);

  ctx->block_size = PAGE_SIZE;
  ctx->rdwt_err = OK;

  buf_list_init();
}

int bdev_open(dev_t dev, int access) {
  return OK;
}

int bdev_close(dev_t dev) {
  return OK;
}

void lmfs_put_block(struct buf *bp, int block_type) {

  int blocksize = lmfs_fs_block_size();
  assert(blocksize == 4096);
  
  if (bp == NULL) {
    return;
  }
  assert(bp->lmfs_bytes == (size_t)blocksize);

  bp->lmfs_count--;
  assert(bp->lmfs_count >= 0);

  if (bp->lmfs_count > 0) {
    return;
  }

  if (!lmfs_isclean(bp)) {
    //assert(pwrite(ctx->fd, bp->data, blocksize, offset) == blocksize);
#ifdef DEBUG
    PNOTICE("write: block %d", bp->lmfs_blocknr);
#endif
    Component::workid_t wid = g_block_layer->async_write(bp->buffer_handle, 0, bp->lmfs_blocknr, 1);
    while(!g_block_layer->check_completion(wid));
   }

  buf_list_remove(bp);

  g_block_layer->free_io_buffer(bp->buffer_handle);
#ifdef DEBUG
  PLOG("freed 4K block (%p)", bp->data);
#endif
  //  free(bp->data);
  
  free(bp);
}

struct buf *lmfs_get_block(dev_t dev, block_t block, int only_search) {
  return lmfs_get_block_ino(dev, block, only_search, VMC_NO_INODE, 0);
}

struct buf *lmfs_get_block_ino(dev_t dev, block_t block, int only_search,
  ino_t ino, u64_t off) {

  int blocksize = lmfs_fs_block_size();
  assert(blocksize == 4096);
  
  char *data;
  struct buf *bp;

  BUF_LIST_FOREACH(bp) {
    if (bp->lmfs_blocknr == block) {
      bp->lmfs_count++;
      return bp;
    }
  }

  assert(lmfs_nr_bufs() > 0);

  bp = (struct buf*)malloc(sizeof(struct buf));

  assert(bp != NULL);
  memset(bp, 0, sizeof(struct buf));
  
  /* allocate space for block */  
  bp->buffer_handle = g_block_layer->allocate_io_buffer(blocksize,
                                                        blocksize,
                                                        Component::NUMA_NODE_ANY);
  data = (char*) g_block_layer->virt_addr(bp->buffer_handle);

  //  assert(pread(ctx->fd, data, blocksize, offset) == blocksize);
#ifdef DEBUG
  PNOTICE("read: block %d",block);
#endif
  Component::workid_t wid = g_block_layer->async_read(bp->buffer_handle, 0, block, 1);
  while(!g_block_layer->check_completion(wid));

  bp->data = data;
  bp->lmfs_blocknr = block;
  bp->lmfs_dev = fs_dev;
  bp->lmfs_count = 1;
  bp->lmfs_bytes = blocksize;
  bp->lmfs_inode = ino;
  bp->lmfs_inode_offset = off;

  buf_list_add(bp);
  return bp;
}

dev_t lmfs_dev(struct buf *bp) {
  return bp->lmfs_dev;
}

void lmfs_markdirty(struct buf *bp) {
  bp->lmfs_flags |= VMMC_DIRTY;
}

void lmfs_markclean(struct buf *bp) {
  bp->lmfs_flags &= ~VMMC_DIRTY;
}

int lmfs_isclean(struct buf *bp) {
  return !(bp->lmfs_flags & VMMC_DIRTY);
}

int lmfs_bytes(struct buf *bp) {
  return bp->lmfs_bytes;
}

void lmfs_flushall(void) {
  struct buf *bp;
  int blocksize = lmfs_fs_block_size();

  BUF_LIST_FOREACH(bp) {
    assert(bp->lmfs_bytes == (size_t)blocksize);

    if (!lmfs_isclean(bp)) {
      //      assert(pwrite(ctx->fd, bp->data, blocksize, offset) == blocksize);
#ifdef DEBUG
      PNOTICE("write (flushall): block %d", bp->lmfs_blocknr);
#endif
      Component::workid_t wid = g_block_layer->async_write(bp->buffer_handle,
                                                           0,
                                                           bp->lmfs_blocknr,
                                                           1);
      while(!g_block_layer->check_completion(wid));

      lmfs_markclean(bp);
    }
  }
}

int lmfs_fs_block_size(void) {
  return ctx->block_size;
}

void lmfs_blockschange(dev_t dev, int delta) {
}

int lmfs_rdwt_err(void) {
  return ctx->rdwt_err;
}

int lmfs_bufs_in_use(void) {
  int i = 0;
  struct buf *bp;

  BUF_LIST_FOREACH(bp) {
    i++;
  }

  return i;
}

int lmfs_nr_bufs(void) {
  return DEFAULT_NR_BUFS - lmfs_bufs_in_use();
}

void lmfs_set_blocksize(int blocksize, int major) {
  assert(buf_list_empty());
  ctx->block_size = blocksize;
}

void lmfs_reset_rdwt_err(void) {
  ctx->rdwt_err = OK;
}

void lmfs_invalidate(dev_t device) {
}

/*
 * Copy data from the caller into the local address space.
 */
int
fsdriver_copyin(const struct fsdriver_data * data, size_t off, void * ptr,
  size_t len)
{

  /* Do nothing for peek requests. */
  if (data == NULL)
    return OK;

  /* The data size field is used only for this integrity check. */
  if (off + len > data->size)
    panic("fsdriver: copy-in buffer overflow");

  memcpy(ptr, &data->ptr[off], len);
  return OK;
}

/*
 * Copy data from the local address space to the caller.
 */
int
fsdriver_copyout(const struct fsdriver_data * data, size_t off,
  const void * ptr, size_t len)
{

  /* Do nothing for peek requests. */
  if (data == NULL)
    return OK;

  /* The data size field is used only for this integrity check. */
  if (off + len > data->size)
    panic("fsdriver: copy-out buffer overflow");

  memcpy(&data->ptr[off], ptr, len);
  return OK;
}

/*
 * Zero out a data region in the caller.
 */
int
fsdriver_zero(const struct fsdriver_data * data, size_t off, size_t len)
{

  /* Do nothing for peek requests. */
  if (data == NULL)
    return OK;

  /* The data size field is used only for this integrity check. */
  if (off + len > data->size)
    panic("fsdriver: copy-out buffer overflow");

  memset(&data->ptr[off], 0, len);
  return OK;
}

/*
 * Initialize a directory entry listing.
 */
void
fsdriver_dentry_init(struct fsdriver_dentry * __restrict dentry,
  const struct fsdriver_data * __restrict data, size_t bytes,
  char * __restrict buf, size_t bufsize)
{

  dentry->data = data;
  dentry->data_size = bytes;
  dentry->data_off = 0;
  dentry->buf = buf;
  dentry->buf_size = bufsize;
  dentry->buf_off = 0;
}

/*
 * Add an entry to a directory entry listing.  Return the entry size if it was
 * added, zero if no more entries could be added and the listing should stop,
 * or an error code in case of an error.
 */
ssize_t
fsdriver_dentry_add(struct fsdriver_dentry * __restrict dentry, ino_t ino_nr,
  const char * __restrict name, size_t namelen, unsigned int type)
{
  struct dirent *dirent;
  size_t len, used;
  int r;

  /* We could do several things here, but it should never happen.. */
  if (namelen > MAXNAMLEN)
    panic("fsdriver: directory entry name excessively long");

  len = _DIRENT_RECLEN(dirent, namelen);

  if (dentry->data_off + dentry->buf_off + len > dentry->data_size) {
    if (dentry->data_off == 0 && dentry->buf_off == 0)
      return -EINVAL;

    return 0;
  }

  if (dentry->buf_off + len > dentry->buf_size) {
    if (dentry->buf_off == 0)
      panic("fsdriver: getdents buffer too small");

    if ((r = fsdriver_copyout(dentry->data, dentry->data_off,
        dentry->buf, dentry->buf_off)) != OK)
      return r;

    dentry->data_off += dentry->buf_off;
    dentry->buf_off = 0;
  }

  dirent = (struct dirent *)&dentry->buf[dentry->buf_off];
  dirent->d_fileno = ino_nr;
  dirent->d_reclen = len;
  dirent->d_namlen = namelen;
  dirent->d_type = type;
  memcpy(dirent->d_name, name, namelen);

  /*
   * Null-terminate the name, and zero out any alignment bytes after it,
   * so as not to leak any data.
   */
  used = _DIRENT_NAMEOFF(dirent) + namelen;
  if (used >= len)
    panic("fsdriver: inconsistency in dirent record");
  memset(&dirent->d_name[namelen], 0, len - used);

  dentry->buf_off += len;

  return len;
}

/*
 * Finish a directory entry listing operation.  Return the total number of
 * bytes copied to the caller, or an error code in case of an error.
 */
ssize_t
fsdriver_dentry_finish(struct fsdriver_dentry *dentry)
{
  int r;

  if (dentry->buf_off > 0) {
    if ((r = fsdriver_copyout(dentry->data, dentry->data_off,
        dentry->buf, dentry->buf_off)) != OK)
      return r;

    dentry->data_off += dentry->buf_off;
  }

  return dentry->data_off;
}

}

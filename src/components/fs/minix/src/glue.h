#ifndef __MFS_GLUE_H__
#define __MFS_GLUE_H__

#include <assert.h>
#include <stdint.h>
//#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/time.h>

typedef uint8_t         u8_t;
typedef uint16_t        u16_t;
typedef uint32_t        u32_t;
typedef uint64_t        u64_t;

typedef int8_t          i8_t;
typedef int16_t         i16_t;
typedef int32_t         i32_t;
typedef int64_t         i64_t;

typedef uint32_t zone_t;      /* zone number */
typedef uint32_t block_t;     /* block number */
typedef uint32_t bit_t;       /* bit number in a bit map */
typedef uint16_t zone1_t;     /* zone number for V1 file systems */
typedef uint32_t bitchunk_t; /* collection of bits in a bitmap */

struct buf {
  /* Data portion of the buffer. */
  void *data;
  uint64_t buffer_handle; /* TODO: may be we set up a slab allocator */
  
  /* Header portion of the buffer - internal to libminixfs. */
  struct buf *lmfs_next;       /* used to link all free bufs in a chain */
  struct buf *lmfs_prev;       /* used to link all free bufs the other way */
  struct buf *lmfs_hash;       /* used to link bufs on hash chains */
  block_t lmfs_blocknr;        /* block number of its (minor) device */
  dev_t lmfs_dev;              /* major | minor device where block resides */
  char lmfs_count;             /* number of users of this buffer */
  char lmfs_needsetcache;      /* to be identified to VM */
  unsigned int lmfs_bytes;     /* Number of bytes allocated in bp */
  u32_t lmfs_flags;            /* Flags shared between VM and FS */

  /* If any, which inode & offset does this block correspond to?
   * If none, VMC_NO_INODE
   */
  ino_t lmfs_inode;
  u64_t lmfs_inode_offset;
};

/* Resulting node properties. */
struct fsdriver_node {
  ino_t fn_ino_nr;    /* inode number */
  mode_t fn_mode;     /* file mode */
  off_t fn_size;      /* file size */
  uid_t fn_uid;     /* owning user ID */
  gid_t fn_gid;     /* owning group ID */
  dev_t fn_dev;     /* device number, for block/char dev */
};

/* Opaque data structure for the fsdriver_copyin, _copyout, _zero functions. */
struct fsdriver_data {
  char *ptr;
  size_t size;
};

/* Opaque data structure for the fsdriver_dentry_ functions. */
struct fsdriver_dentry {
  const struct fsdriver_data *data;
  size_t data_size;
  size_t data_off;
  char *buf;
  size_t buf_size;
  size_t buf_off;
};

#define MAXNAMLEN       511
#define IFTODT(mode)    (((mode) & 0170000) >> 12)

#define _DIRENT_ALIGN(dp) (sizeof((dp)->d_fileno) - 1)

#define _DIRENT_NAMEOFF(dp) __builtin_offsetof(__typeof__(*(dp)), d_name)

#define _DIRENT_RECLEN(dp, namlen) \
    ((_DIRENT_NAMEOFF(dp) + (namlen) + 1 + _DIRENT_ALIGN(dp)) & \
    ~_DIRENT_ALIGN(dp))

struct dirent {
  ino_t d_fileno;
  uint16_t d_reclen;
  uint16_t d_namlen;
  uint8_t  d_type;
  char    d_name[MAXNAMLEN + 1];
};

#define EXTERN        extern    /* used in *.h files */
#define __unused        __attribute__((__unused__))

#define OK 0

#define TRUE               1    /* used for turning integers into Booleans */
#define FALSE              0    /* used for turning integers into Booleans */

#define NO_BLOCK              ((block_t) 0) /* absence of a block number */
#define NO_ENTRY                ((ino_t) 0) /* absence of a dir entry */
#define NO_ZONE                ((zone_t) 0) /* absence of a zone number */
#define NO_DEV                  ((dev_t) 0) /* absence of a device numb */
#define NO_LINK         ((nlink_t) 0) /* absence of incoming links */

#define BYTE            0377  /* mask for 8 bits */
#define READING            0    /* copy data to user */
#define WRITING            1    /* copy data from user */
#define PEEKING            2    /* retrieve FS data without copying */

#define INODE_BLOCK        0                             /* inode block */
#define DIRECTORY_BLOCK    1                             /* directory block */
#define INDIRECT_BLOCK     2                             /* pointer block */
#define MAP_BLOCK          3                             /* bit map */
#define FULL_DATA_BLOCK    5                             /* data, fully used */
#define PARTIAL_DATA_BLOCK 6                             /* data, partly used*/

/* get_block arguments */
#define NORMAL             0    /* forces get_block to do disk read */
#define NO_READ            1    /* prevents get_block from doing disk read */
#define PREFETCH           2    /* tells get_block not to read or mark dev */

/* Flag bits for i_mode in the inode. */
#define I_TYPE          0170000 /* this field gives inode type */
#define I_UNIX_SOCKET   0140000 /* unix domain socket */
#define I_SYMBOLIC_LINK 0120000 /* file is a symbolic link */
#define I_REGULAR       0100000 /* regular file, not dir or special */
#define I_BLOCK_SPECIAL 0060000 /* block special file */
#define I_DIRECTORY     0040000 /* file is a directory */
#define I_CHAR_SPECIAL  0020000 /* character special file */
#define I_NAMED_PIPE    0010000 /* named pipe (FIFO) */
#define I_SET_UID_BIT   0004000 /* set effective uid_t on exec */
#define I_SET_GID_BIT   0002000 /* set effective gid_t on exec */
#define I_SET_STCKY_BIT 0001000 /* sticky bit */
#define ALL_MODES       0007777 /* all bits for user, group and others */
#define RWX_MODES       0000777 /* mode bits for RWX only */
#define R_BIT           0000004 /* Rwx protection bit */
#define W_BIT           0000002 /* rWx protection bit */
#define X_BIT           0000001 /* rwX protection bit */
#define I_NOT_ALLOC     0000000 /* this inode is free */

#define LINK_MAX    32767 /* max file link count */

#define FSC_READ  0   /* read or bread call */
#define FSC_WRITE 1   /* write or bwrite call */
#define FSC_PEEK  2   /* peek or bpeek call */

#define FSC_UNLINK  0   /* unlink call */
#define FSC_RMDIR 1   /* rmdir call */

/* VFS/FS flags */
#define REQ_RDONLY    001 /* FS is mounted read-only */
#define REQ_ISROOT    002 /* FS is root file system */

/* Bits in 'BDEV_ACCESS' field of block device open requests. */
#  define BDEV_R_BIT    0x01  /* open with read access */
#  define BDEV_W_BIT    0x02  /* open with write access */

#define RES_NOFLAGS   000
#define RES_THREADED    001 /* FS supports multithreading */
#define RES_HASPEEK   002 /* FS implements REQ_PEEK/REQ_BPEEK */
#define RES_64BIT   004 /* FS can handle 64-bit file sizes */

#define END_OF_FILE   (-104)        /* eof detected */

/* flags for vm cache functions */
#define VMMC_FLAGS_LOCKED       0x01    /* someone is updating the flags; don't read/write */
#define VMMC_DIRTY              0x02    /* dirty buffer and it may not be evicted */
#define VMMC_EVICTED            0x04    /* VM has evicted the buffer and it's invalid */
#define VMMC_BLOCK_LOCKED       0x08    /* client is using it and it may not be evicted */

/* special inode number for vm cache functions */
#define VMC_NO_INODE    0 /* to reference a disk block, no associated file */

#define NR_IOREQS     64    /* maximum number of entries in an iorequest */

#define PAGE_SIZE 0x1000

#define ASSERT assert
#define rounddown(x,y)  (((x)/(y))*(y))

#ifndef UTIME_NOW
#define UTIME_NOW  0x3fffffff
#endif

#ifndef UTIME_OMIT
#define UTIME_OMIT 0x3ffffffe
#endif

static inline unsigned long ex64lo(u64_t i)
{
        return (unsigned long)i;
}

static inline unsigned long ex64hi(u64_t i)
{
        return (unsigned long)(i>>32);
}

static inline u64_t make64(unsigned long lo, unsigned long hi)
{
        return ((u64_t)hi << 32) | (u64_t)lo;
}

void glue_setup();

void panic(const char *fmt, ...);
void util_stacktrace(void);
time_t clock_time(struct timespec *tv);

int bdev_open(dev_t dev, int access);
int bdev_close(dev_t dev);

void lmfs_put_block(struct buf *bp, int block_type);
struct buf *lmfs_get_block(dev_t dev, block_t block, int only_search);
struct buf *lmfs_get_block_ino(dev_t dev, block_t block, int only_search,
  ino_t ino, u64_t off);
dev_t lmfs_dev(struct buf *bp);
void lmfs_markdirty(struct buf *bp);
void lmfs_markclean(struct buf *bp);
int lmfs_isclean(struct buf *bp);
int lmfs_bytes(struct buf *bp);
void lmfs_flushall(void);
int lmfs_fs_block_size(void);
void lmfs_blockschange(dev_t dev, int delta);
int lmfs_rdwt_err(void);
int lmfs_bufs_in_use(void);
int lmfs_nr_bufs(void);
void lmfs_set_blocksize(int blocksize, int major);
void lmfs_reset_rdwt_err(void);
void lmfs_invalidate(dev_t device);

int fsdriver_copyout(const struct fsdriver_data *data, size_t off,
        const void *ptr, size_t len);
int fsdriver_copyin(const struct fsdriver_data *data, size_t off, void *ptr,
        size_t len);
int fsdriver_zero(const struct fsdriver_data *data, size_t off, size_t len);
void fsdriver_dentry_init(struct fsdriver_dentry * __restrict dentry,
        const struct fsdriver_data * __restrict data, size_t bytes,
        char * __restrict buf, size_t bufsize);
ssize_t fsdriver_dentry_add(struct fsdriver_dentry * __restrict dentry,
        ino_t ino_nr, const char * __restrict name, size_t namelen,
        unsigned int type);
ssize_t fsdriver_dentry_finish(struct fsdriver_dentry *dentry);

#endif

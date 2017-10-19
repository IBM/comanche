#include <common/logging.h>
#include "fs.h"
#include "buf.h"
#include "inode.h"
#include <string.h>
#include <stdlib.h>
#include <libgen.h>

#define FUSE_USE_VERSION 29
#include <fuse.h>
#define PACKAGE_NAME    "fuse-mfs"
#define PACKAGE_VERSION "0.4"

enum {
  KEY_VERSION,
  KEY_HELP,
};

static struct fsdriver_node root_node;
static const char *device_path = NULL;

static int next_name(const char *ptr, char *name, size_t size, const char **nextp) {
  size_t i;

  while(*ptr == '/') {
    ptr++;
  }

  if (*ptr == '\0') {
    *nextp = NULL;
    return OK;
  }

  for (i = 0; i < size && ptr[i] != '\0' && ptr[i] != '/'; ++i) {
    name[i] = ptr[i];
  }

  if (i == size) {
    return -ENAMETOOLONG;
  }
  name[i] = '\0';

  *nextp = &ptr[i];
  return OK;
}

static int lookup(ino_t root_nr, const char *path, struct fsdriver_node *node) {
  ino_t dir_nr;
  int r, is_mountpt;
  const char *ptr, *nextp;
  char name[MFS_NAME_MAX + 1];
  struct fsdriver_node cur_node;

  if (path[0] != '/') {
    return -ENOENT;
  }
  int rc = fs_lookup(root_nr, ".", &cur_node, &is_mountpt);
  if(rc != OK) return rc;

  ptr = path;
  while (1) {
    r = next_name(ptr, name, sizeof(name), &nextp);
    if (r != OK) {
      return r;
    }

    if (nextp == NULL) {
      *node = cur_node;
      return OK;
    }

    if (is_mountpt && !strcmp(name, "..")) {
      fs_putnode(cur_node.fn_ino_nr, 1);
      return -EINVAL;
    }

    ptr = nextp;

    dir_nr = cur_node.fn_ino_nr;
    r = fs_lookup(dir_nr, name, &cur_node, &is_mountpt);
    fs_putnode(dir_nr, 1);
    if (r != OK) {
      return r;
    }

    if (!S_ISDIR(cur_node.fn_mode)) {
      r = next_name(ptr, name, sizeof(name), &nextp);

      if (r != OK) {
        return r;
      }

      if (nextp == NULL) {
        *node = cur_node;
        return OK;
      } else {
        fs_putnode(cur_node.fn_ino_nr, 1);
        return -ENOTDIR;
      }
    }
  }
}

static void setup(void) {
  int i;

  /* Init inode table */
  for (i = 0; i < NR_INODES; ++i) {
    inode[i].i_count = 0;
    cch[i] = 0;
  }
  init_inode_cache();

  glue_setup();
}

static int minix_getattr(const char *path, struct stat *stbuf) {
  int r;
  struct fsdriver_node node;

  r = lookup(root_node.fn_ino_nr, path, &node);
  if (r != OK) {
    return r;
  }

  r = fs_stat(node.fn_ino_nr, stbuf);
  fs_putnode(node.fn_ino_nr, 1);

  return r;
}



static int minix_open(const char *path, struct fuse_file_info *fi) {
  struct fsdriver_node node;
  return lookup(root_node.fn_ino_nr, path, &node);
}

static int minix_release(const char *path, struct fuse_file_info *fi) {
  int r;
  struct fsdriver_node node;

  r = lookup(root_node.fn_ino_nr, path, &node);
  if (r != OK) {
    return r;
  }

  fs_putnode(node.fn_ino_nr, 2);
  return OK;
}

static int minix_read(const char *path, char *buf, size_t size,
                      off_t offset, struct fuse_file_info *fi) {
  int r;
  ssize_t nread;
  struct fsdriver_node node;
  struct fsdriver_data data = {
    .ptr  = buf,
    .size = size,
  };

  r = lookup(root_node.fn_ino_nr, path, &node);
  if (r != OK) {
    return r;
  }

  nread = fs_readwrite(node.fn_ino_nr, &data, data.size, offset, FSC_READ);
  fs_putnode(node.fn_ino_nr, 1);

  return nread;
}

static int minix_write(const char *path, const char *buf, size_t size,
                       off_t offset, struct fuse_file_info *fi) {
  int r;
  ssize_t nread;
  struct fsdriver_node node;
  struct fsdriver_data data = {
    .ptr  = (char*)buf,
    .size = size,
  };

  r = lookup(root_node.fn_ino_nr, path, &node);
  if (r != OK) {
    return r;
  }

  nread = fs_readwrite(node.fn_ino_nr, &data, data.size, offset, FSC_WRITE);
  fs_putnode(node.fn_ino_nr, 1);

  return nread;
}

static int minix_readdir(const char *path, void *buf, fuse_fill_dir_t filler,
                         off_t offset, struct fuse_file_info *fi) {
  int i, r;
  char dbuf[PAGE_SIZE], *ptr = dbuf;
  ssize_t nread;
  struct fsdriver_node node;
  struct fsdriver_data data = {
    .ptr  = dbuf,
    .size = sizeof(dbuf),
  };
  struct dirent *dp;
  off_t pos = 0;

  r = lookup(root_node.fn_ino_nr, path, &node);
  if (r != OK) {
    return r;
  }

  while (1) {
    i = 0;
    nread = fs_getdents(node.fn_ino_nr, &data, data.size, &pos);

    if (nread <= 0) {
      fs_putnode(node.fn_ino_nr, 1);
      return nread;
    }

    while (i < nread) {
      dp = (struct dirent*)&ptr[i];
      filler(buf, dp->d_name, NULL, 0);
      i += dp->d_reclen;
    }
  }
}

static int minix_create(const char *path, mode_t mode, struct fuse_file_info *fi) {
  int r;
  off_t dir_nr;
  struct fsdriver_node node;
  struct fuse_context *context = fuse_get_context();
  char *cp;

  cp = strdup(path);
  r = lookup(root_node.fn_ino_nr, dirname(cp), &node);
  free(cp);

  if (r != OK) {
    return r;
  }

  cp = strdup(path);
  dir_nr = node.fn_ino_nr;
  r = fs_create(dir_nr, basename(cp), I_REGULAR | mode, context->uid, context->gid, &node);
  fs_putnode(dir_nr, 1);
  free(cp);

  return r;
}

static int minix_link(const char *from, const char *to) {
  int r;
  struct fsdriver_node from_node, to_dir_node;
  char *cp;

  r = lookup(root_node.fn_ino_nr, from, &from_node);
  if (r != OK) {
    return r;
  }

  cp = strdup(to);
  r = lookup(root_node.fn_ino_nr, dirname(cp), &to_dir_node);
  free(cp);
  if (r != OK) {
    fs_putnode(from_node.fn_ino_nr, 1);
    return r;
  }

  cp = strdup(to);
  r = fs_link(to_dir_node.fn_ino_nr, basename(cp), from_node.fn_ino_nr);
  fs_putnode(from_node.fn_ino_nr, 1);
  fs_putnode(to_dir_node.fn_ino_nr, 1);
  free(cp);

  return r;
}

static int minix_symlink(const char *from, const char *to) {
  int r;
  struct fsdriver_node node;
  struct fsdriver_data data = {
    .ptr  = (char*)from,
    .size = strlen(from),
  };
  struct fuse_context *context = fuse_get_context();
  char *cp;

  cp = strdup(to);
  r = lookup(root_node.fn_ino_nr, dirname(cp), &node);
  free(cp);
  if (r != OK) {
    return r;
  }

  cp = strdup(to);
  r = fs_slink(node.fn_ino_nr, basename(cp), context->uid, context->gid, &data, data.size);
  fs_putnode(node.fn_ino_nr, 1);
  free(cp);

  return r;
}

static int minix_readlink(const char *path, char *buf, size_t size) {
  int r;
  ssize_t nread;
  struct fsdriver_node node;
  struct fsdriver_data data = {
    .ptr  = buf,
    .size = size,
  };

  r = lookup(root_node.fn_ino_nr, path, &node);
  if (r != OK) {
    return r;
  }

  fsdriver_zero(&data, 0, data.size);
  nread = fs_rdlink(node.fn_ino_nr, &data, data.size);
  fs_putnode(node.fn_ino_nr, 1);

  return (nread > 0 ? OK : nread);
}

static int minix_rename(const char *from, const char *to) {
  int r;
  struct fsdriver_node node, from_node, to_node;
  char *from_cp, *to_cp;
  struct inode *rip;

  from_cp = strdup(from);
  r = lookup(root_node.fn_ino_nr, dirname(from_cp), &from_node);
  free(from_cp);
  if (r != OK) {
    return r;
  }

  to_cp = strdup(to);
  r = lookup(root_node.fn_ino_nr, dirname(to_cp), &to_node);
  free(to_cp);
  if (r != OK) {
    fs_putnode(from_node.fn_ino_nr, 1);
    return r;
  }

  from_cp = strdup(from);
  to_cp = strdup(to);

  r = fs_rename(from_node.fn_ino_nr, basename(from_cp),
                to_node.fn_ino_nr, basename(to_cp));

  fs_putnode(from_node.fn_ino_nr, 1);
  fs_putnode(to_node.fn_ino_nr, 1);
  free(from_cp);
  free(to_cp);

  if (r == OK && lookup(root_node.fn_ino_nr, to, &node) == OK) {
    rip = get_inode(fs_dev, node.fn_ino_nr);

    if (rip) {
      rip->i_update |= CTIME;
      IN_MARKDIRTY(rip);
      put_inode(rip);
    }

    fs_putnode(node.fn_ino_nr, 1);
  }

  return r;
}

static int minix_mkdir(const char *path, mode_t mode) {
  int r;
  struct fsdriver_node node;
  struct fuse_context *context = fuse_get_context();
  char *cp;

  cp = strdup(path);
  r = lookup(root_node.fn_ino_nr, dirname(cp), &node);
  free(cp);

  if (r != OK) {
    return r;
  }

  cp = strdup(path);
  r = fs_mkdir(node.fn_ino_nr, basename(cp), I_DIRECTORY | mode, context->uid, context->gid);
  fs_putnode(node.fn_ino_nr, 1);
  free(cp);

  return r;
}

static int minix_unlink(const char *path) {
  int r;
  struct fsdriver_node node;
  char *cp;

  cp = strdup(path);
  r = lookup(root_node.fn_ino_nr, dirname(cp), &node);
  free(cp);

  if (r != OK) {
    return r;
  }

  cp = strdup(path);
  r = fs_unlink(node.fn_ino_nr, basename(cp), FSC_UNLINK);
  fs_putnode(node.fn_ino_nr, 1);
  free(cp);

  return r;
}

static int minix_rmdir(const char *path) {
  int r;
  struct fsdriver_node node;
  char *cp;

  cp = strdup(path);
  r = lookup(root_node.fn_ino_nr, dirname(cp), &node);
  free(cp);

  if (r != OK) {
    return r;
  }

  cp = strdup(path);
  r = fs_unlink(node.fn_ino_nr, basename(cp), FSC_RMDIR);
  fs_putnode(node.fn_ino_nr, 1);
  free(cp);

  return r;
}

static int minix_utimens(const char *path, const struct timespec tv[2]) {
  int r;
  struct fsdriver_node node;
  struct timespec atime = tv[0], mtime = tv[1];

  r = lookup(root_node.fn_ino_nr, path, &node);
  if (r != OK) {
    return r;
  }

  r = fs_utime(node.fn_ino_nr, &atime, &mtime);
  fs_putnode(node.fn_ino_nr, 1);

  return r;
}

static int minix_truncate(const char *path, off_t size) {
  int r;
  struct fsdriver_node node;

  r = lookup(root_node.fn_ino_nr, path, &node);
  if (r != OK) {
    return r;
  }

  r = fs_trunc(node.fn_ino_nr, size, 0);
  fs_putnode(node.fn_ino_nr, 1);

  return r;
}

static int minix_chmod(const char *path, mode_t mode) {
  int r;
  struct fsdriver_node node;

  r = lookup(root_node.fn_ino_nr, path, &node);
  if (r != OK) {
    return r;
  }

  r = fs_chmod(node.fn_ino_nr, &mode);
  fs_putnode(node.fn_ino_nr, 1);

  return r;
}

static int minix_chown(const char *path, uid_t uid, gid_t gid) {
  int r;
  struct fsdriver_node node;
  mode_t mode;
  struct inode *rip;

  r = lookup(root_node.fn_ino_nr, path, &node);
  if (r != OK) {
    return r;
  }

  if ((rip = get_inode(fs_dev, node.fn_ino_nr)) == NULL) {
    fs_putnode(node.fn_ino_nr, 1);
    return(-EINVAL);
  }
  uid = (uid == (uid_t)-1 ? rip->i_uid : uid);
  gid = (gid == (gid_t)-1 ? rip->i_gid : gid);
  put_inode(rip);

  r = fs_chown(node.fn_ino_nr, uid, gid, &mode);
  fs_putnode(node.fn_ino_nr, 1);

  return r;
}

static int minix_mknod(const char *path, mode_t mode, dev_t dev) {
  int r;
  off_t dir_nr;
  struct fsdriver_node node;
  struct fuse_context *context = fuse_get_context();
  char *cp;

  cp = strdup(path);
  r = lookup(root_node.fn_ino_nr, dirname(cp), &node);
  free(cp);

  if (r != OK) {
    return r;
  }

  cp = strdup(path);
  dir_nr = node.fn_ino_nr;
  r = fs_mknod(dir_nr, basename(cp), mode, context->uid, context->gid, dev);
  fs_putnode(dir_nr, 1);
  free(cp);

  return r;
}

static void usage(const char *progname) {
  printf(
"usage: %s device mountpoint [options]\n"
"\n"
"general options:\n"
"    -o opt,[opt...]        mount options\n"
"    -h   --help            print help\n"
"    -V   --version         print version\n"
"\n", progname);
}

static int minix_opt_proc(void *data, const char *arg, int key,
                          struct fuse_args *outargs) {
  switch (key) {
    case FUSE_OPT_KEY_NONOPT:
      if (device_path == NULL) {
        device_path = strdup(arg);
        return 0;
      }
      return 1;

    case KEY_VERSION:
      printf("%s version %s\n", PACKAGE_NAME, PACKAGE_VERSION);
      fuse_opt_add_arg(outargs, "--version");
      fuse_main(outargs->argc, outargs->argv, NULL, NULL);
      exit(0);

    case KEY_HELP:
      usage(outargs->argv[0]);
      fuse_opt_add_arg(outargs, "-ho");
      fuse_main(outargs->argc, outargs->argv, NULL, NULL);
      exit(1);

    default:
      return 1;
  }
}

static struct fuse_operations minix_ops = {
  .getattr    = minix_getattr,
  .readdir    = minix_readdir,
  .open       = minix_open,
  .release    = minix_release,
  .opendir    = minix_open,
  .releasedir = minix_release,
  .write      = minix_write,
  .read       = minix_read,
  .create     = minix_create,
  .mkdir      = minix_mkdir,
  .mknod      = minix_mknod,
  .link       = minix_link,
  .unlink     = minix_unlink,
  .symlink    = minix_symlink,
  .readlink   = minix_readlink,
  .rename     = minix_rename,
  .rmdir      = minix_rmdir,
  .utimens    = minix_utimens,
  .truncate   = minix_truncate,
  .chmod      = minix_chmod,
  .chown      = minix_chown,
};

static struct fuse_opt minix_opts[] = {
  FUSE_OPT_KEY("-V",        KEY_VERSION),
  FUSE_OPT_KEY("--version", KEY_VERSION),
  FUSE_OPT_KEY("-h",        KEY_HELP),
  FUSE_OPT_KEY("--help",    KEY_HELP),
  FUSE_OPT_END,
};

int do_run_fuse() {
  unsigned int res_flags;

  dev_t dev = makedev(10, 1);
  struct fuse_args args = FUSE_ARGS_INIT(0, NULL);


  /* disable multi-threaded operation */

  /* if (fuse_opt_parse(&args, NULL, minix_opts, minix_opt_proc) == -1) {  */
  /*   panic("couldn't parse the argument");  */
  /* }  */

  fuse_opt_add_arg(&args, "-f");
  fuse_opt_add_arg(&args, "-d");
  fuse_opt_add_arg(&args, "-s");
  fuse_opt_add_arg(&args, "/tmp/foo");

  setup();

  if (fs_mount(dev, 0, &root_node, &res_flags) != OK) {
    panic("couldn't mount the device errno=%d", errno);
  }

  fs_mountpt(root_node.fn_ino_nr);
  fuse_main(args.argc, args.argv, &minix_ops, NULL);
  fs_unmount();

  return 0;
}


void init_minix_fs(void)
{
  setup();
}

#if 0
/* SEF functions and variables. */
static void sef_local_startup(void);
static int sef_cb_init_fresh(int type, sef_init_info_t *info);
static void sef_cb_signal_handler(int signo);

/*===========================================================================*
 *				main                                         *
 *===========================================================================*/
int main(int argc, char *argv[])
{
/* This is the main routine of this service. */

  /* SEF local startup. */
  env_setargs(argc, argv);
  sef_local_startup();

  /* The fsdriver library does the actual work here. */
  fsdriver_task(&mfs_table);

  return(0);
}

/*===========================================================================*
 *			       sef_local_startup			     *
 *===========================================================================*/
static void sef_local_startup()
{
  /* Register init callbacks. */
  sef_setcb_init_fresh(sef_cb_init_fresh);
  sef_setcb_init_restart(sef_cb_init_fail);

  /* No live update support for now. */

  /* Register signal callbacks. */
  sef_setcb_signal_handler(sef_cb_signal_handler);

  /* Let SEF perform startup. */
  sef_startup();
}

/*===========================================================================*
 *		            sef_cb_init_fresh                                *
 *===========================================================================*/
static int sef_cb_init_fresh(int UNUSED(type), sef_init_info_t *UNUSED(info))
{
/* Initialize the Minix file server. */
  int i;

  lmfs_may_use_vmcache(1);

  /* Init inode table */
  for (i = 0; i < NR_INODES; ++i) {
	inode[i].i_count = 0;
	cch[i] = 0;
  }

  init_inode_cache();

  lmfs_buf_pool(DEFAULT_NR_BUFS);

  return(OK);
}

/*===========================================================================*
 *		           sef_cb_signal_handler                             *
 *===========================================================================*/
static void sef_cb_signal_handler(int signo)
{
  /* Only check for termination signal, ignore anything else. */
  if (signo != SIGTERM) return;

  fs_sync();

  fsdriver_terminate();
}
#endif

#if 0
/*===========================================================================*
 *				cch_check				     *
 *===========================================================================*/
static void cch_check(void)
{
  int i;

  for (i = 0; i < NR_INODES; ++i) {
	if (inode[i].i_count != cch[i] && req_nr != REQ_GETNODE &&
	    req_nr != REQ_PUTNODE && req_nr != REQ_READSUPER &&
	    req_nr != REQ_MOUNTPOINT && req_nr != REQ_UNMOUNT &&
	    req_nr != REQ_SYNC && req_nr != REQ_LOOKUP) {
		PLOG("MFS(%d) inode(%lu) cc: %d req_nr: %d\n", sef_self(),
			inode[i].i_num, inode[i].i_count - cch[i], req_nr);
	}

	cch[i] = inode[i].i_count;
  }
}
#endif


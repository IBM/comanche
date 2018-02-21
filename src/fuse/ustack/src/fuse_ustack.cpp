/*
  FUSE: Filesystem in Userspace wrapper for Ustack-Store
*/
#define FUSE_USE_VERSION 26
//#define USE_NVME_DEVICE
#define DB_LOCATION "/home/danielwaddington/comanche/src/fuse/ustack"

#include <fuse.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <unistd.h>
#include <assert.h>
#include <sys/xattr.h>
#include <common/str_utils.h>
#include <core/dpdk.h>
#include <rapidjson/document.h>

#include <api/components.h>
#include <api/store_itf.h>
#include <api/block_itf.h>
#include <api/blob_itf.h>

#include "ustack.h"

using namespace Component;
/*
 * Command line options
 *
 * We can't set default values for the char* fields here because
 * fuse_opt_parse would attempt to free() them when the user specifies
 * different values on the command line.
 */
static struct options {
  const char *filename;
  const char *contents;
  int show_help;
} options;

#define OPTION(t, p)                            \
  { t, offsetof(struct options, p), 1 }

static const struct fuse_opt option_spec[] = {
  OPTION("--name=%s", filename),
  OPTION("--contents=%s", contents),
  OPTION("-h", show_help),
  OPTION("--help", show_help),
  FUSE_OPT_END
};


Ustack * ustack;


class File_handle
{
public:
  File_handle(const std::string id, Component::IBlob * store) :
    _store(store), _id(id) {
    assert(store);
    _iob = ustack->block->allocate_io_buffer(MB(32),KB(4),NUMA_NODE_ANY);
    _iob_virt = ustack->block->virt_addr(_iob);
    //    _cursor = _store->open(id);
  }

  ~File_handle() {
    ustack->block->free_io_buffer(_iob);
  }

  void* read(uint64_t offset, size_t size) {
    // _store->get(_id,
    //             _iob,
    //             0,
    //             0/*queue*/);
    assert(0);
    return _iob_virt;
  }
    
  void write(const char * data, size_t size, uint64_t offset) {
    memcpy(_iob_virt, data, size);
    // _store->put(_id,
    //             "none",
    //             _iob,
    //             offset,
    //             0/*queue*/);
  }


  Component::io_buffer_t& iob() { return _iob; }

  inline void * buffer() const { return _iob_virt; }

  size_t size;
  
private:
  Component::IBlob::cursor_t _cursor;
  const std::string      _id;
  Component::IBlob *    _store;
  Component::io_buffer_t _iob;
  void *                 _iob_virt;
};
  

static void *fuse_ustack_init(struct fuse_conn_info *conn)
{
  DPDK::eal_init(1024);
  PLOG("fuse_ustack: DPDK init OK.");

  ustack = new Ustack();
  
  return NULL;
}

static void fuse_ustack_destroy(void *private_data)
{
  TRACE();

  delete ustack;
}

static int fuse_ustack_getattr(const char *path, struct stat *st)
{
  int res = 0;
  memset(st, 0, sizeof(struct stat));
  st->st_uid = getuid();
	st->st_gid = getgid();
	st->st_atime = time( NULL );
	st->st_mtime = time( NULL );
  
  PLOG("getattr: path=[%s]", path);
  if(strcmp(path,"/") == 0) {
    st->st_mode = S_IFDIR | 0755;
		st->st_nlink = 2;
    return res;
  }

  size_t entity_size = 0;
  if(ustack->store->check_key(path+1, entity_size) == 0) {
    PLOG("no entity");
    return -ENOENT;
  }
  
  st->st_mode = S_IFREG | 0444;
  st->st_nlink = 2; /* number of hard links */
  st->st_size = entity_size;

  return res;
}

static int fuse_ustack_create(const char * filename,
                              mode_t mode,
                              struct fuse_file_info * fi)
{
  TRACE();
  PLOG("create: %s", filename);
  //  ustack->store->put(filename, "none", nullptr, MB(8));
  return 0;
}

static int fuse_ustack_unlink(const char * filename)
{
  PLOG("unlink: %s", filename);
  return 0;
}

static int fuse_ustack_readdir(const char *path,
                               void *buf,
                               fuse_fill_dir_t filler,
                               off_t offset,
                               struct fuse_file_info *fi)
{
  using namespace rapidjson;
  TRACE();
  
  (void) offset;
  (void) fi;
  
  filler(buf, ".", NULL, 0);
  filler(buf, "..", NULL, 0);

  std::vector<std::string> mdv;
  ustack->store->get_metadata_vector("{\"id\":\".*\"}", mdv);

  //  auto last_record = ustack->store->get_record_count();
  //  for(uint64_t r=1;r<=last_record ;r++) {
  for(auto& m : mdv) {
    Document document;
    document.Parse(m.c_str());
    const char * id  = document["id"].GetString();
    filler(buf, id, NULL, 0);
  }
  
  // //  filler(buf, options.filename, NULL, 0);
  // if(filler(buf, "foobar", NULL, 0) == 1) return offset;
  // off++;
  
  return 0;
}

static int fuse_ustack_open(const char *path, struct fuse_file_info *fi)
{
  TRACE();

  size_t size = 0;
  if(ustack->store->check_key(path, size) == 0)
    return -ENOENT;

  File_handle * fh;

  try {
    fh = new File_handle(std::string(path), ustack->store);
    fi->fh = reinterpret_cast<uint64_t>(fh);
    fh->size = size;
  }
  catch(General_exception excp) {
    return -ENOENT;
  }
  PLOG("open OK!");
  // if (strcmp(path+1, options.filename) != 0)
    
  // if ((fi->flags & O_ACCMODE) != O_RDONLY)
  //   return -EACCES;

  return 0;
}

static int fuse_ustack_read(const char *path,
                            char *buf,
                            size_t size,
                            off_t offset,
                            struct fuse_file_info *fi)
{
  PLOG("read: path=%s size=%lu offset=%lu", path, size, offset);
  size_t len;

  size_t entity_size = 0;
  if(ustack->store->check_key(path, entity_size) == 0)
    return -ENOENT;
  
  assert(fi->fh);

  File_handle * fh = reinterpret_cast<File_handle*>(fi->fh);
  fh->read(offset, size);
  memcpy(buf, fh->buffer(), size); /* perform memory copy for the moment */

  return size;
}

static int fuse_ustack_access(const char *, int)
{
  return 0;
}

static int fuse_ustack_flush(const char *, struct fuse_file_info *)
{
  return 0;
}

static int fuse_ustack_release(const char* path, struct fuse_file_info *fi)
{
  PLOG("release (%s)", path);
  return 0;  
}

static int fuse_ustack_setxattr(const char* path,
                                const char* name,
                                const char* value,
                                size_t size,
                                int flags)
{
  PLOG("setxttr (%s,%s,%s)", path, name, value);
  return -1;
}


static int fuse_ustack_getxattr(const char * path,
                                const char *name,
                                char * value, size_t size)
{
  PLOG("getxttr (%s,%s,%s)", path, name, value);
  //  memset(value, 0, size);
  return -ENODATA;
}


static int fuse_ustack_truncate(const char *, off_t)
{
  TRACE();
  return 0;
}

static int fuse_ustack_utimens(const char* path, const timespec*)
{
  TRACE();
  return 0;
}
static int fuse_ustack_fallocate(const char * path, int, off_t, off_t, struct fuse_file_info *)
{
  TRACE();
  return 0;
}

static int fuse_ustack_write(const char* path, const char *buf, size_t size, off_t offset, struct fuse_file_info* fi)
{
  PLOG("write: (%s, %lu, %lu)", path, size, offset);
  File_handle * fh = reinterpret_cast<File_handle*>(fi->fh);
  fh->write(buf, size, offset);
  
  return 0;
}

/** 
 * End of fuse hooks -----------------------------------------------------------
 * 
 */





static void show_help(const char *progname)
{
  printf("usage: %s [options] <mountpoint>\n\n", progname);
  printf("File-system specific options:\n"
         "    --name=<s>          Name of the \"fuse_ustack\" file\n"
         "                        (default: \"fuse_ustack\")\n"
         "    --contents=<s>      Contents \"fuse_ustack\" file\n"
         "                        (default \"Fuse_Ustack, World!\\n\")\n"
         "\n");
}


int main(int argc, char *argv[])
{
  struct fuse_args args = FUSE_ARGS_INIT(argc, argv);
  /* Set defaults -- we have to use strdup so that
     fuse_opt_parse can free the defaults if other
     values are specified */
  options.filename = strdup("fuse_ustack");
  options.contents = strdup("Ustack World!\n");
  /* Parse options */
  if (fuse_opt_parse(&args, &options, option_spec, NULL) == -1)
    return 1;
  /* When --help is specified, first print our own file-system
     specific help text, then signal fuse_main to show
     additional help (by adding `--help` to the options again)
     without usage: line (by setting argv[0] to the empty
     string) */
  if (options.show_help) {
    show_help(argv[0]);
    assert(fuse_opt_add_arg(&args, "--help") == 0);
    args.argv[0] = (char*) "";
  }
  
  static struct fuse_operations fuse_ustack_oper;
  fuse_ustack_oper.init    = fuse_ustack_init;
  fuse_ustack_oper.destroy = fuse_ustack_destroy;
  fuse_ustack_oper.getattr = fuse_ustack_getattr;
  fuse_ustack_oper.readdir = fuse_ustack_readdir;
  fuse_ustack_oper.open    = fuse_ustack_open;
  fuse_ustack_oper.read    = fuse_ustack_read;
  fuse_ustack_oper.write    = fuse_ustack_write;
  fuse_ustack_oper.access  = fuse_ustack_access;
  fuse_ustack_oper.flush   = fuse_ustack_flush;
  fuse_ustack_oper.create   = fuse_ustack_create;
  fuse_ustack_oper.unlink   = fuse_ustack_unlink;
  fuse_ustack_oper.getxattr = fuse_ustack_getxattr;
  fuse_ustack_oper.setxattr = fuse_ustack_setxattr;
  fuse_ustack_oper.release = fuse_ustack_release;
  fuse_ustack_oper.truncate = fuse_ustack_truncate;
  fuse_ustack_oper.utimens = fuse_ustack_utimens;
  fuse_ustack_oper.fallocate = fuse_ustack_fallocate;
  return fuse_main(args.argc, args.argv, &fuse_ustack_oper, NULL);
}

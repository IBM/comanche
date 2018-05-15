/*
  FUSE: Filesystem in Userspace wrapper for Append-Store
*/
#define FUSE_USE_VERSION 26
//#define USE_NVME_DEVICE
#define DB_LOCATION "/home/danielwaddington/comanche/src/fuse/append"

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

#include <api/components.h>
#include <api/store_itf.h>


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

static IBlock_device * create_nvme_block_device(const std::string device_name);
static IBlock_device * create_posix_block_device(const std::string path);
static IStore * create_append_store(IBlock_device * block);

struct {
  IBlock_device * block = nullptr;
  IStore *        store = nullptr;
} g_state;

class File_handle
{
public:
  File_handle(const std::string id, Component::IStore * store) :
    _store(store), _id(id) {
    assert(store);
    _iob = g_state.block->allocate_io_buffer(MB(32),KB(4),NUMA_NODE_ANY);
    _iob_virt = g_state.block->virt_addr(_iob);
  }

  ~File_handle() {
    g_state.block->free_io_buffer(_iob);
  }

  void* read(uint64_t offset, size_t size) {
    _store->get(_id,
                _iob,
                0,
                0/*queue*/);
    return _iob_virt;
  }
    
  void write(const char * data, size_t size, uint64_t offset) {
    memcpy(_iob_virt, data, size);
    _store->put(_id,
                "none",
                _iob,
                offset,
                0/*queue*/);
  }


  Component::io_buffer_t& iob() { return _iob; }

  inline void * buffer() const { return _iob_virt; }

private:
  const std::string      _id;
  Component::IStore *    _store;
  Component::io_buffer_t _iob;
  void *                 _iob_virt;
};
  

static void *fuse_append_init(struct fuse_conn_info *conn)
{

  DPDK::eal_init(1024);
  PLOG("fuse_append: DPDK init OK.");

#ifdef USE_NVME_DEVICE
  g_state.block = create_nvme_block_device("01:00.0");
#else
  g_state.block = create_posix_block_device("./block.dat");
#endif

  PLOG("fuse_append: block device %p", g_state.block);
    
  g_state.store = create_append_store(g_state.block);
  assert(g_state.store);
  
  return NULL;
}

static void fuse_append_destroy(void *private_data)
{
  TRACE();
  g_state.store->release_ref();
  g_state.block->release_ref();
}

static int fuse_append_getattr(const char *path, struct stat *stbuf)
{
  int res = 0;
  memset(stbuf, 0, sizeof(struct stat));

  if(g_state.store->check_path(path) == 0)
    return -ENOENT;

  stbuf->st_mode = S_IFREG | 0444;
  stbuf->st_nlink = 1;
  stbuf->st_size = strlen(options.contents);

  return res;
}

static int fuse_append_create(const char * filename,
                              mode_t mode,
                              struct fuse_file_info * fi)
{
  PLOG("create: %s", filename);
  g_state.store->put(filename, "none", nullptr, MB(8));
  return 0;
}

static int fuse_append_unlink(const char * filename)
{
  PLOG("unlink: %s", filename);
  return 0;
}

static int fuse_append_readdir(const char *path,
                               void *buf,
                               fuse_fill_dir_t filler,
                               off_t offset,
                               struct fuse_file_info *fi)
{
  TRACE();
  
  (void) offset;
  (void) fi;
  
  filler(buf, ".", NULL, 0);
  filler(buf, "..", NULL, 0);

  auto last_record = g_state.store->get_record_count();
  for(uint64_t r=1;r<=last_record ;r++) {
    std::string md = g_state.store->get_metadata(r);
    filler(buf, md.c_str(), NULL, 0);
  }
  
  // //  filler(buf, options.filename, NULL, 0);
  // if(filler(buf, "foobar", NULL, 0) == 1) return offset;
  // off++;
  
  return 0;
}

static int fuse_append_open(const char *path, struct fuse_file_info *fi)
{
  TRACE();

  if(g_state.store->check_path(path) == 0)
    return -ENOENT;

  File_handle * fh;

  try {
    fh = new File_handle(std::string(path), g_state.store);
    fi->fh = reinterpret_cast<uint64_t>(fh);
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

static int fuse_append_read(const char *path,
                            char *buf,
                            size_t size,
                            off_t offset,
                            struct fuse_file_info *fi)
{
  PLOG("read: path=%s size=%lu offset=%lu", path, size, offset);
  size_t len;

  if(g_state.store->check_path(path) == 0)
    return -ENOENT;
  
  assert(fi->fh);

  File_handle * fh = reinterpret_cast<File_handle*>(fi->fh);
  fh->read(offset, size);
  memcpy(buf, fh->buffer(), size); /* perform memory copy for the moment */

  return size;
}

static int fuse_append_access(const char *, int)
{
  return 0;
}

static int fuse_append_flush(const char *, struct fuse_file_info *)
{
  return 0;
}

static int fuse_append_release(const char* path, struct fuse_file_info *fi)
{
  PLOG("release (%s)", path);
  return 0;  
}

static int fuse_append_setxattr(const char* path,
                                const char* name,
                                const char* value,
                                size_t size,
                                int flags)
{
  PLOG("setxttr (%s,%s,%s)", path, name, value);
  return -1;
}


static int fuse_append_getxattr(const char * path,
                                const char *name,
                                char * value, size_t size)
{
  PLOG("getxttr (%s,%s,%s)", path, name, value);
  //  memset(value, 0, size);
  return -ENODATA;
}


static int fuse_append_truncate(const char *, off_t)
{
  TRACE();
  return 0;
}

static int fuse_append_utimens(const char* path, const timespec*)
{
  TRACE();
  return 0;
}
static int fuse_append_fallocate(const char * path, int, off_t, off_t, struct fuse_file_info *)
{
  TRACE();
  return 0;
}

static int fuse_append_write(const char* path, const char *buf, size_t size, off_t offset, struct fuse_file_info* fi)
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


static IBlock_device * create_nvme_block_device(const std::string device_name)
{
  PLOG("creating nvme block device: %s", device_name.c_str());
  
  IBase * comp = load_component("libcomanche-blknvme.so",
                                block_nvme_factory);
  
  assert(comp);
  IBlock_device_factory * fact = (IBlock_device_factory *)
    comp->query_interface(IBlock_device_factory::iid());
  
  cpu_mask_t cpus;
  cpus.add_core(2);
  
  auto block = fact->create(device_name.c_str(), &cpus);
  assert(block);
  fact->release_ref();

  return block;
}

static IBlock_device * create_posix_block_device(const std::string path)
{
  IBase * comp = load_component("libcomanche-blkposix.so",
                                                      block_posix_factory);
  assert(comp);
  PLOG("Block_device factory loaded OK.");
  
  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());
  std::string config_string;
  config_string = "{\"path\":\"";
  //  config_string += "/dev/nvme0n1";1
  config_string += path; //"./blockfile.dat";
  //  config_string += "\"}";
  config_string += "\",\"size_in_blocks\":80000}";

  auto block = fact->create(config_string);
  assert(block);
  fact->release_ref();
  return block;
}

static IStore * create_append_store(IBlock_device * block)
{
  assert(block);
  
  IBase * comp = load_component("libcomanche-storeappend.so",
                                Component::store_append_factory);
  assert(comp);
  IStore_factory * fact = (IStore_factory *) comp->query_interface(IStore_factory::iid());
  auto store = fact->create(getlogin(), "teststore", DB_LOCATION, block, 0); //FLAGS_FORMAT);  
  fact->release_ref();

  // test put
  // {
  //   std::string data = Common::random_string(128);
  //   store->put(Common::random_string(8), "metadata", (void*) data.c_str(), data.length());
  // }

  return store;
}



static void show_help(const char *progname)
{
  printf("usage: %s [options] <mountpoint>\n\n", progname);
  printf("File-system specific options:\n"
         "    --name=<s>          Name of the \"fuse_append\" file\n"
         "                        (default: \"fuse_append\")\n"
         "    --contents=<s>      Contents \"fuse_append\" file\n"
         "                        (default \"Fuse_Append, World!\\n\")\n"
         "\n");
}


int main(int argc, char *argv[])
{
  struct fuse_args args = FUSE_ARGS_INIT(argc, argv);
  /* Set defaults -- we have to use strdup so that
     fuse_opt_parse can free the defaults if other
     values are specified */
  options.filename = strdup("fuse_append");
  options.contents = strdup("Fuse_Append World!\n");
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
  
  static struct fuse_operations fuse_append_oper;
  fuse_append_oper.init    = fuse_append_init;
  fuse_append_oper.destroy = fuse_append_destroy;
  fuse_append_oper.getattr = fuse_append_getattr;
  fuse_append_oper.readdir = fuse_append_readdir;
  fuse_append_oper.open    = fuse_append_open;
  fuse_append_oper.read    = fuse_append_read;
  fuse_append_oper.write    = fuse_append_write;
  fuse_append_oper.access  = fuse_append_access;
  fuse_append_oper.flush   = fuse_append_flush;
  fuse_append_oper.create   = fuse_append_create;
  fuse_append_oper.unlink   = fuse_append_unlink;
  fuse_append_oper.getxattr = fuse_append_getxattr;
  fuse_append_oper.setxattr = fuse_append_setxattr;
  fuse_append_oper.release = fuse_append_release;
  fuse_append_oper.truncate = fuse_append_truncate;
  fuse_append_oper.utimens = fuse_append_utimens;
  fuse_append_oper.fallocate = fuse_append_fallocate;
  return fuse_main(args.argc, args.argv, &fuse_append_oper, NULL);
}

/*
 * (C) Copyright IBM Corporation 2018. All rights reserved.
 *
 */

/* 
 * Authors: 
 * 
 * Feng Li (fengggli@yahoo.com)
 */

/*
 * Fuse integration with ustack.
 *
 * This is the slow path. e.g. creation/open/close the file.
 * read/write operations are send from ustack channel
 *
 */

/*
 * File system abstraction
 *
 * each object will be a file in the mountdir/
 * TODO: add namespace, such the object name "ns1/item1" will be put into the corresponding subdir
 */

#define FUSE_USE_VERSION 26

#include <fuse.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>


#include <common/logging.h>
#include <core/dpdk.h>
#include <api/components.h>
#include <api/kvstore_itf.h>

#include "ustack.h"
#include "ustack_client_ioctl.h"
#include "kv_ustack_info.h"

#define PMSTORE_PATH "libcomanche-pmstore.so"
#define FILESTORE_PATH "libcomanche-storefile.so"
#define NVMESTORE_PATH "libcomanche-nvmestore.so"

/**
 * ustack: the userspace zero copy communiation mechenism.
 *
 * Client can also direct issue posix file operations.
 */
Ustack *_ustack;

/**
 * Initialize filesystem
 *
 * This will just create/open a kvstore instance 
 * fuse-api: The return value will passed in the private_data field of
 * fuse_context to all file operations and as a parameter to the
 * destroy() method.
 *
 */
void * kvfs_ustack_init (struct fuse_conn_info *conn){
  (void)conn; // TODO: component owner, name  can be in the fuse option

  Component::IKVStore *store;
  Component::IBase * comp; 

  //std::string component("filestore");
  std::string component("nvmestore");

  if(component == "pmstore") {
    comp = Component::load_component(PMSTORE_PATH, Component::pmstore_factory);
  }
  else if(component == "filestore") {
    comp = Component::load_component(FILESTORE_PATH, Component::filestore_factory);
  }
  else if(component == "nvmestore") {
    comp = Component::load_component(NVMESTORE_PATH, Component::nvmestore_factory);
  }
  else throw General_exception("unknown --component option (%s)", component.c_str());

  assert(comp);

  Component::IKVStore_factory * fact = (Component::IKVStore_factory *) comp->query_interface(Component::IKVStore_factory::iid());

  std::map<std::string, std::string> params;
  params["owner"] = "testowner";
  params["name"] = "testname";
  params["pci"] = "20:00.0";
  params["pm_path"] = "/mnt/pmem0/";
  params["persist_type"] = "hstore";
  // params["persist_type"] = "filestore";

  unsigned debug_level = 0;

  store = fact->create(debug_level, params);

  fact->release_ref();

  PINF("[%s]: fs loaded using component %s ", __func__, component.c_str());

  // KV_ustack_info * info = new KV_ustack_info("owner", "name", store);
  KV_ustack_info * info = new KV_ustack_info_cached("owner", "name", store);

  // init ustack and start accepting connections
  std::string ustack_name = "ipc:///tmp//kv-ustack.ipc";
  // DPDK::eal_init(1024);

  _ustack = new Ustack(ustack_name.c_str(), info);
  return info;
}

/**
 * Clean up filesystem
 *
 * this will close a kvstore
 * fuse-api: alled on filesystem exit.
 */
void kvfs_ustack_destroy (void *user_data){
  KV_ustack_info *info = reinterpret_cast<KV_ustack_info *>(user_data);

  Component::IKVStore *store = info->get_store();
  store->release_ref();
  
  delete info;
  delete _ustack;
};


/**
 * Create and open a file
 *
 * If the file does not exist, first create it with the specified
 * mode, and then open it.
 *
 * If this method is not implemented or under Linux kernel
 * versions earlier than 2.6.15, the mknod() and open() methods
 * will be called instead.
 *
 * Introduced in version 2.5
 */
int kvfs_ustack_create (const char *path, mode_t mode, struct fuse_file_info * fi){
  uint64_t handle;
  //PLOG("create: %s", filename);

  KV_ustack_info *info = reinterpret_cast<KV_ustack_info *>(fuse_get_context()->private_data);

  //unsigned long ino = (fuse_get_context()->fuse->ctr); // this require lowlevel api

  handle = info->alloc_id();
  assert(handle);

  PDBG("[%s]: create new file(%s)!",__func__, path);

  info->insert_item(handle, path+1);
  assert(S_OK == info->open_file(handle));
  fi->fh = handle;
  PLOG("[%s]: create entry No.%lu: key(%s)", __func__, handle, path+1);
  return 0;
}

/** Remove a file */
int kvfs_ustack_unlink(const char *path){
  KV_ustack_info *info = reinterpret_cast<KV_ustack_info *>(fuse_get_context()->private_data);

  uint64_t handle = info->get_id(path+1);
  /** remove obj from pool*/
  if(S_OK != info->remove_item(handle)){
    PWRN("%s remove fuse_fh(%lu) failed", __func__, handle);
  }

  return 0;
}

static int kvfs_ustack_getattr(const char *path, struct stat *stbuf)
{
	int res = 0;

  KV_ustack_info *info = reinterpret_cast<KV_ustack_info *>(fuse_get_context()->private_data);

  uint64_t handle = info->get_id(path+1);

	memset(stbuf, 0, sizeof(struct stat));
	if (strcmp(path, "/") == 0) {
		stbuf->st_mode = S_IFDIR | 0755;
		stbuf->st_nlink = 2;
	}else if(handle){
		stbuf->st_mode = S_IFREG | 0444;
		stbuf->st_nlink = 1;
    PDBG("[%s]: get attr!",__func__);
		stbuf->st_size = info->get_item_size(handle);
  }
  else{
    PWRN("[%s]: path(%s) cannot found!",__func__, path);
		res = -ENOENT;
  }

	return res;
}

static int kvfs_ustack_readdir(const char *path, void *buf, fuse_fill_dir_t filler,
			 off_t offset, struct fuse_file_info *fi)
{
	(void) offset;
	(void) fi;

  KV_ustack_info *info = reinterpret_cast<KV_ustack_info *>(fuse_get_context()->private_data);

  filler(buf, ".", NULL, 0);
  filler(buf, "..", NULL, 0);

	if (strcmp(path, "/") == 0){
    std::unordered_map<uint64_t,  std::string> all_items;
    all_items = info->get_all_items();
    for(const auto &item : all_items){
      PINF("item name: %s", item.second.c_str());
      filler(buf, item.second.c_str(), NULL, 0);
    }
  }else{
    return -1;
  }
  PINF("pool name all iterated");

	//filler(buf, kvfs_ustack_path + 1, NULL, 0);
	return 0;
}


static int kvfs_ustack_open(const char *path, struct fuse_file_info *fi)
{
  uint64_t handle;
  //PLOG("create: %s", filename);

  KV_ustack_info *info = reinterpret_cast<KV_ustack_info *>(fuse_get_context()->private_data);


  //TODO: check access: return -EACCES;
  handle = info->get_id(path+1);
  if(handle){
    PDBG("[%s]: open existing file (%s)!",__func__, path);
    fi->fh = handle;
    assert(S_OK == info->open_file(handle));
    return 0;
  }
  else{
    PINF("[%s]: open not found exsiting!",__func__);
    return -ENOENT;
  }
}


/** Release an open file
 *
 * Release is called when there are no more references to an open file
 */
int kvfs_ustack_release(const char *path, struct fuse_file_info *fi){
  //PLOG("create: %s", filename);

  KV_ustack_info *info = reinterpret_cast<KV_ustack_info *>(fuse_get_context()->private_data);

  //TODO: check access: return -EACCES;
  uint64_t handle = fi->fh;
  
  if(handle){
    PDBG("[%s]: closing file!",__func__);
    assert(S_OK == info->close_file(handle));
    return 0;
  }
  else{
    PWRN("[%s]: file not opened!",__func__);
    return -ENOENT;
  }
}

int kvfs_ustack_fallocate (const char *path, int mode, off_t offset, off_t length,
      struct fuse_file_info *){

    PWRN("[%s]: fallocate doesn't do anything ", __func__);
    return 0;
}

int kvfs_ustack_flush(const char *, struct fuse_file_info *){
    PWRN("[%s]: flush doesn't do anything ", __func__);
    return 0;
}

/* 
 * read and write are called if you don't use the file operations from the ustack_client.
 * e.g. run "echo "hello world" > ./mymount/tmp.data"
 */
static int kvfs_ustack_read(const char *path, char *buf, size_t size, off_t offset,
		      struct fuse_file_info *fi)
{
  (void) offset;
	(void) fi;


  KV_ustack_info *info = reinterpret_cast<KV_ustack_info *>(fuse_get_context()->private_data);
  if(S_OK!=info->read(fi->fh, buf , size)){
    PERR("[%s]: read error", __func__);
    return -1;
  }

  PLOG("[%s]: read value %s from path %s ", __func__, buf, path);

  return size;
}

/** Write data to an open file
 * 
 * Write should return exactly the number of bytes requested
 * except on error.	 An exception to this is when the 'direct_io'
 * mount option is specified (see read operation).
 *
 * Changed in version 2.2
 */
int (kvfs_ustack_write) (const char *path, const char *buf, size_t size, off_t offset, struct fuse_file_info *fi){
  (void)offset;

  uint64_t id;

  PLOG("[%s]: write content %s to path %s", __func__,buf, path);

  KV_ustack_info *info = reinterpret_cast<KV_ustack_info *>(fuse_get_context()->private_data);

  id = fi->fh;
  info->write(id, buf, size);

  info->set_item_size(id, size);
  return size;
}

static int kvfs_ustack_ioctl(const char *path, int cmd, void *arg,
		      struct fuse_file_info *fi, unsigned int flags, void *data)
{
	(void) arg;

	/*if (fioc_file_type(path) != FIOC_FILE)*/
		/*return -EINVAL;*/

	if (flags & FUSE_IOCTL_COMPAT)
		return -ENOSYS;

	switch (cmd) {
	case USTACK_GET_FUSE_FH:
		*(uint64_t *)data = fi->fh;
		return 0;
	}

	return -EINVAL;
}

int main(int argc, char *argv[])
{
  static struct fuse_operations oper;
  memset(&oper, 0, sizeof(struct fuse_operations));
	oper.getattr	= kvfs_ustack_getattr;
	oper.readdir	= kvfs_ustack_readdir;
	oper.open		= kvfs_ustack_open;
	oper.read		= kvfs_ustack_read;
	oper.write		= kvfs_ustack_write;
  oper.init = kvfs_ustack_init;
  oper.create = kvfs_ustack_create;
  oper.destroy = kvfs_ustack_destroy;
  oper.ioctl = kvfs_ustack_ioctl;
  oper.unlink = kvfs_ustack_unlink;
  oper.fallocate = kvfs_ustack_fallocate;
  oper.flush = kvfs_ustack_flush;
  oper.release = kvfs_ustack_release;

	return fuse_main(argc, argv, &oper, NULL);
}

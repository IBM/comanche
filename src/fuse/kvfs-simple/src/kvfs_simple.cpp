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
 * Simple fuse integration of ikv interface.
 *
 * Here "simple" means it uses default fuse interfaces for all file operations, 
 * which bounce between kernel/user space.
 * This can be improved greatly by intercept some of the intensive operations and handle them completel in userspace (as it's done in fuse/ustack)
 *
 * This is modified from the  hello example in the libfuse source tree.
 */

/*
 * File system abstraction
 *
 * each pool will be a folder in the mount root, each object will be a file
 */

#define FUSE_USE_VERSION 26

#include <fuse.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>

#include <api/components.h>
#include <api/kvstore_itf.h>

#define PMSTORE_PATH "libcomanche-pmstore.so"
#define FILESTORE_PATH "libcomanche-storefile.so"
#define NVMESTORE_PATH "libcomanche-nvmestore.so"

static const char *kvfs_simple_str = "Hello World!\n";
static const char *kvfs_simple_path = "/hello";

/**
 * Initialize filesystem
 *
 * This will just create/open a kvstore instance 
 * fuse-api: The return value will passed in the private_data field of
 * fuse_context to all file operations and as a parameter to the
 * destroy() method.
 *
 */
void * kvfs_simple_init (struct fuse_conn_info *conn){
  (void)conn; // TODO: component owner, name  can be in the fuse option

  Component::IKVStore *store;
  Component::IBase * comp; 

  std::string component("filestore");

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

  store = fact->create("owner","name");
  fact->release_ref();

  PINF("[%s]: fs loaded using component %s ", __func__, component.c_str());
  return store;
}

/**
 * Clean up filesystem
 *
 * this will close a kvstore
 * fuse-api: alled on filesystem exit.
 */
void kvfs_simple_destroy (void *user_data){
  Component::IKVStore *store = reinterpret_cast<Component::IKVStore *>(user_data);
  store->release_ref();
  
};

//static int open(); 
//static int create(); //allocate and get_direct? do we need write_direct
//static  mkdir // ikv-create_pool

/* read and write must operate on /pool/key */
//static int read(); // pool->ikv-get
//static int write(); // pool->ikv-put


static int kvfs_simple_getattr(const char *path, struct stat *stbuf)
{
	int res = 0;

	memset(stbuf, 0, sizeof(struct stat));
	if (strcmp(path, "/") == 0) {
		stbuf->st_mode = S_IFDIR | 0755;
		stbuf->st_nlink = 2;
	} else if (strcmp(path, kvfs_simple_path) == 0) {
		stbuf->st_mode = S_IFREG | 0444;
		stbuf->st_nlink = 1;
		stbuf->st_size = strlen(kvfs_simple_str);
	} else
		res = -ENOENT;

	return res;
}

static int kvfs_simple_readdir(const char *path, void *buf, fuse_fill_dir_t filler,
			 off_t offset, struct fuse_file_info *fi)
{
	(void) offset;
	(void) fi;

  filler(buf, ".", NULL, 0);
  filler(buf, "..", NULL, 0);

	if (strcmp(path, "/") == 0){
    // show pools

  }else{
    // show the objects in the pool
  }

  // this should give all the pool
	//filler(buf, kvfs_simple_path + 1, NULL, 0);
	return 0;
}

static int kvfs_simple_open(const char *path, struct fuse_file_info *fi)
{
	if (strcmp(path, kvfs_simple_path) != 0)
		return -ENOENT;

	if ((fi->flags & 3) != O_RDONLY)
		return -EACCES;

	return 0;
}

static int kvfs_simple_read(const char *path, char *buf, size_t size, off_t offset,
		      struct fuse_file_info *fi)
{
	size_t len;
	(void) fi;
	if(strcmp(path, kvfs_simple_path) != 0)
		return -ENOENT;

	len = strlen(kvfs_simple_str);
	if (offset < len) {
		if (offset + size > len)
			size = len - offset;
		memcpy(buf, kvfs_simple_str + offset, size);
	} else
		size = 0;

	return size;
}



int main(int argc, char *argv[])
{
  static struct fuse_operations oper;
  memset(&oper, 0, sizeof(struct fuse_operations));
	oper.getattr	= kvfs_simple_getattr;
	oper.readdir	= kvfs_simple_readdir;
	oper.open		= kvfs_simple_open;
	oper.read		= kvfs_simple_read;
  oper.init = kvfs_simple_init;
  oper.destroy = kvfs_simple_destroy;

	return fuse_main(argc, argv, &oper, NULL);
}

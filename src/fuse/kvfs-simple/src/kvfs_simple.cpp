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
 * This can be improved greatly by intercept some of the intensive operations and handle them complete in userspace (as it's done in fuse/ustack)
 *
 * This is modified from the  hello example in the libfuse source tree.
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

#include <unordered_map>

#include <common/logging.h>
#include <api/components.h>
#include <api/kvstore_itf.h>

#define PMSTORE_PATH "libcomanche-pmstore.so"
#define FILESTORE_PATH "libcomanche-storefile.so"
#define NVMESTORE_PATH "libcomanche-nvmestore.so"

using pool_t     = uint64_t;

/*
 * private information for this mounting, this only modifies metadata
 */
class Mount_info{
  public:
    struct File_meta{
      size_t size;
    };

    Mount_info(const std::string owner, const std::string name, Component::IKVStore *store)
      :_owner(owner), _name(name), _store(store), _asigned_ids(0){

      _pool = _store->create_pool("/mnt/pmem0", "pool-kvfs-simple", MB(12));
    }

    ~Mount_info(){
    }

    Component::IKVStore *get_store(){
      return _store;
    }

    std::unordered_map<uint64_t, std::string> get_all_items(){
      return _items;
    }; 

    /*
     * get a available file id
     */
    uint64_t alloc_id(){
      return ++_asigned_ids;
    }

    uint64_t insert_item(uint64_t id, std::string key){
      _items.insert(std::pair<uint64_t, std::string>(id, key));
    }

    /*
     * look up this file based on the filename
     *
     * @return 0 if not found
     */
    uint64_t get_id(std::string item){
      for(const auto & i : _items){
        if(i.second == item)
          return i.first;
      }
      return 0;
    }

    /* get the file size 
     */
    size_t get_item_size(uint64_t id){
      return _file_meta[id].size;
    }

    void set_item_size(uint64_t id, size_t size){
      _file_meta[id].size = size;
    }

    status_t write(uint64_t id, const void * value, size_t size){
      const std::string key = _items[id];
      return _store->put(_pool, key, value, size);
    }

    status_t read(uint64_t id, void * value, size_t size){
      const std::string key = _items[id];
      void * tmp;
      size_t rd_size;
      // tmp will be redirected
      if(S_OK != _store->get(_pool, key, tmp, rd_size)){
        return -1;
      }
      memcpy(value, tmp, size);
      free(tmp);
      return S_OK;
    }


  private:
      //ownership of this store
      std::string _owner;      
      std::string _name;
      Component::IKVStore *_store;
      pool_t _pool;

      std::unordered_map<uint64_t, std::string> _items; // file id and key TODO: put the key to the File_meta!
      std::unordered_map<uint64_t, File_meta> _file_meta;  
      std::atomic<uint64_t> _asigned_ids; //current assigned ids, to identify each file
};

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

  Mount_info * info = new Mount_info("owner", "name", store);
  return info;
}

/**
 * Clean up filesystem
 *
 * this will close a kvstore
 * fuse-api: alled on filesystem exit.
 */
void kvfs_simple_destroy (void *user_data){
  Mount_info *info = reinterpret_cast<Mount_info *>(user_data);

  Component::IKVStore *store = info->get_store();
  store->release_ref();
  
  delete info;
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
int kvfs_simple_create (const char *path, mode_t mode, struct fuse_file_info * fi){
  uint64_t handle;
  PLOG("create: %s", filename);

  Mount_info *info = reinterpret_cast<Mount_info *>(fuse_get_context()->private_data);

  handle = info->alloc_id();
  assert(handle);

  info->insert_item(handle, path+1);
  fi->fh = handle;
  PINF("[%s]: create entry No. %lu: key %s", __func__, handle, path+1);
  return 0;
}



//static  mkdir // ikv-create_pool

/* read and write must operate on /pool/key */
//static int read(); // pool->ikv-get
//static int write(); // pool->ikv-put


/** Get file attributes.
 *
 * API: Similar to stat().  The 'st_dev' and 'st_blksize' fields are
 * ignored.	 The 'st_ino' field is ignored except if the 'use_ino'
 * mount option is given.
 */

static int kvfs_simple_getattr(const char *path, struct stat *stbuf)
{
	int res = 0;

  Mount_info *info = reinterpret_cast<Mount_info *>(fuse_get_context()->private_data);

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
    PINF("[%s]: open not found exsiting!",__func__);
		res = -ENOENT;
  }

	return res;
}

static int kvfs_simple_readdir(const char *path, void *buf, fuse_fill_dir_t filler,
			 off_t offset, struct fuse_file_info *fi)
{
	(void) offset;
	(void) fi;

  Mount_info *info = reinterpret_cast<Mount_info *>(fuse_get_context()->private_data);

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

	//filler(buf, kvfs_simple_path + 1, NULL, 0);
	return 0;
}


static int kvfs_simple_open(const char *path, struct fuse_file_info *fi)
{
  uint64_t handle;
  PLOG("create: %s", filename);

  Mount_info *info = reinterpret_cast<Mount_info *>(fuse_get_context()->private_data);


  //TODO: check access: return -EACCES;
  handle = info->get_id(path+1);
  if(handle){
    PINF("[%s]: open found exsiting!",__func__);
    fi->fh = handle;
    return 0;
  }
  else{
    PINF("[%s]: open not found exsiting!",__func__);
    return -ENOENT;
  }
}


static int kvfs_simple_read(const char *path, char *buf, size_t size, off_t offset,
		      struct fuse_file_info *fi)
{
  (void) offset;
	(void) fi;


  Mount_info *info = reinterpret_cast<Mount_info *>(fuse_get_context()->private_data);
  if(S_OK!=info->read(fi->fh, buf , size)){
    PERR("[%s]: read error", __func__);
    return -1;
  }

  PINF("[%s]: read value %s from path %s ", __func__, buf, path);

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
int (kvfs_simple_write) (const char *path, const char *buf, size_t size, off_t offset, struct fuse_file_info *fi){
  (void)offset;

  uint64_t id;

  PINF("[%s]: write content %s to path %s", __func__,buf, path);

  Mount_info *info = reinterpret_cast<Mount_info *>(fuse_get_context()->private_data);

  id = fi->fh;
  info->write(id, buf, size);

  info->set_item_size(id, size);
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
	oper.write		= kvfs_simple_write;
  oper.init = kvfs_simple_init;
  oper.create = kvfs_simple_create;
  oper.destroy = kvfs_simple_destroy;

	return fuse_main(argc, argv, &oper, NULL);
}

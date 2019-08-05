/*
 * Description:
 *
 * First created: 2018 Jul 11
 * Last modified: 2018 Jul 12
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#ifndef KV_USTACK_INFO_H_
#define KV_USTACK_INFO_H_

#include <unordered_map>

#include <queue>
#include <api/components.h>
#include <api/kvstore_itf.h>

using namespace Component;

// using kv_ustack_info_t = KV_ustack_info;
class KV_ustack_info;
class KV_ustack_info_cached;

/** Daemon information, used by both ustack handler and fuse daemon*/
using kv_ustack_info_t = KV_ustack_info_cached;

#include "ustack.h"
/*
 * private information for this mounting.
 * Allocate the fuse daemon internal fh; interact with kvstore
 */
class KV_ustack_info{
  public:
    using pool_t     = uint64_t;
    using fuse_fd_t = uint64_t;

    struct File_meta{
      File_meta() = delete;
      File_meta(std::string name): name(name), size(0){}
      ~File_meta(){}

      std::string name;
      size_t size;
    };


    KV_ustack_info(const std::string ustack_name, const std::string owner, const std::string name, Component::IKVStore *store)
      :_owner(owner), _name(name), _store(store), _asigned_ids(0){


      // TODO: why hardcopied size?
      _pool = _store->create_pool("pool-kvfs-simple", MB(12));
    }

    virtual ~KV_ustack_info(){
    }

    Component::IKVStore *get_store(){
      return _store;
    }

    /* All files in this mount*/
    std::vector<std::string> get_filenames(){
      std::vector<std::string> filenames;
      for(auto item = _items.begin(); item!= _items.end(); item++){
        filenames.push_back(item->second->name);
      }
      return filenames;
    }; 

    fuse_fd_t insert_item(std::string key){
      fuse_fd_t id=  ++_asigned_ids;

      _items.insert(std::make_pair(id, new File_meta(key)));
      return id;
    }

    status_t remove_item(uint64_t id){
      status_t ret;
      std::string key =  _items.at(id)->name;

      ret = _store->erase(_pool, key);
      delete _items.at(id);
      _items.erase(id);
      --_asigned_ids;
      return ret;
    }

    /*
     * look up this file based on the filename
     *
     * @return 0 if not found
     */
    fuse_fd_t get_id(std::string item){
      for(const auto & i : _items){
        if(i.second->name == item)
          return i.first;
      }
      return 0;
    }

    /* get the file size 
     */
    size_t get_item_size(fuse_fd_t id){
      return _items.at(id)->size;
    }

    void set_item_size(fuse_fd_t id, size_t size){
      _items.at(id)->size = size;
    }

    /**
     * kvstore interaction
     */
    status_t open_file(fuse_fd_t) {return S_OK;};
    status_t close_file(fuse_fd_t) {return S_OK;};

    status_t write(fuse_fd_t id, const void * value, size_t size, size_t file_offset = 0 ){
      if(file_offset){
        throw General_exception("kv_ustack_basic doesn't support offset");
      }
      const std::string key = _items.at(id)->name;
      assert(value !=  NULL);
      return _store->put(_pool, key, value, size);
    }

    status_t read(fuse_fd_t id, void * value, size_t size, size_t file_offset = 0){
      const std::string key = _items.at(id)->name;
      void * tmp;
      size_t rd_size;
      if(file_offset){
        throw General_exception("kv_ustack_basic doesn't support offset");
      }

      // tmp will be redirected
      if(S_OK != _store->get(_pool, key, tmp, rd_size)){
        return -1;
      }
      memcpy(value, tmp, size);
      free(tmp);
      return S_OK;
    }

  protected:
      // ownership of this store
      std::string _owner;      
      std::string _name;

      Component::IKVStore *_store;
      pool_t _pool;
      std::map<fuse_fd_t, File_meta *> _items;  

  private:

      std::atomic<fuse_fd_t> _asigned_ids; //current assigned ids, to identify each file

};

struct page_cache_entry{
  size_t pg_offset;
  bool is_dirty;

  void * vaddr = nullptr; /** pointer to this locked object*/
  IKVStore::key_t locked_key = nullptr;

  // page_cache_entry() = delete;
  page_cache_entry(size_t pg_offset = 0): pg_offset(pg_offset), is_dirty(false){}
};

/** Use lock/unlock to provide page cache*/
class KV_ustack_info_cached{

  using pool_t     = uint64_t;
  using fuse_fd_t = uint64_t;
  static constexpr size_t k_pool_size = MB(256); /** To save all objs in this mount*/
  static constexpr size_t PAGE_CACHE_SIZE = MB(2);

  public:

    /** Per-file structure*/
    struct File_meta{
      File_meta() = delete;
      File_meta(std::string name, IKVStore * store, IKVStore::pool_t pool): name(name), _store(store), _pool(pool){}

      ~File_meta(){}


      status_t populate_cache(){
        PDBG("cached open called");
        page_cache_entry * entry = new page_cache_entry();
        std::string obj_key = name + "#seg-0";
        size_t out_value_len;

        // If same file is opened again, contents shall be the same
        if(S_OK!= _store->lock(_pool, obj_key, IKVStore::STORE_LOCK_WRITE, entry->vaddr, out_value_len, entry->locked_key)){
          throw General_exception("lock file cached failed");
          }

        PDBG("%s: get locked_key 0x%lx", __func__,  uint64_t(entry->locked_key));
        _cached_pages.push(entry);
        return S_OK;
      }

      /* TODO: I can only flush one time!*/
      status_t flush_cache(){
        while(! _cached_pages.empty()){
          auto entry = _cached_pages.front();
          PDBG("%s: trying to use locked_key 0x%lx", __func__,  uint64_t(entry->locked_key));
          _store->unlock(_pool, entry->locked_key);
          _cached_pages.pop();
        }
        return S_OK;
      }

      std::string name;
      size_t size = 0;

      std::queue<page_cache_entry *> _cached_pages; /** just an array of locked regions*/

      private:
      IKVStore * _store;
      IKVStore::pool_t _pool;
    }; // end of File_meta

    KV_ustack_info_cached(const std::string ustack_name, const std::string owner, const std::string name, Component::IKVStore *store): 
      _owner(owner), _name(name), _store(store), _assigned_ids(0){
      _pool = _store->create_pool(name, k_pool_size);
      if(IKVStore::POOL_ERROR == _pool){
        throw General_exception("%s: initial pool creation failed", __func__);
      }
    };

    ~KV_ustack_info_cached(){
      _store->close_pool(_pool);
    }

    Component::IKVStore *get_store(){
      return _store;
    }

    /* All files in this mount*/
    std::vector<std::string> get_filenames(){
      std::vector<std::string> filenames;
      for(auto item = _items.begin(); item!= _items.end(); item++){
        filenames.push_back(item->second->name);
      }
      return filenames;
    }; 

    fuse_fd_t insert_item(std::string key){
      fuse_fd_t id=  ++_assigned_ids;

      assert(_pool);
      _items.insert(std::make_pair(id, new File_meta(key, _store, _pool)));
      return id;
    }

    status_t remove_item(uint64_t id){
      status_t ret;
      std::string key =  _items.at(id)->name;

      ret = _store->erase(_pool, key);
      delete _items.at(id);
      _items.erase(id);
      --_assigned_ids;
      return ret;
    }

    /*
     * look up this file based on the filename
     *
     * @return 0 if not found
     */
    fuse_fd_t get_id(std::string item){
      for(const auto & i : _items){
        if(i.second->name == item)
          return i.first;
      }
      return 0;
    }

    /* get the file size 
     */
    size_t get_item_size(fuse_fd_t id){
      return _items.at(id)->size;
    }

    void set_item_size(fuse_fd_t id, size_t size){
      _items.at(id)->size = size;
    }


    status_t open_file(fuse_fd_t id)
    {
      File_meta* fileinfo = _items.at(id);
      fileinfo->populate_cache();
      return S_OK;
    }


    /* This will unlock the obj, thus writing persist data to pool*/
    status_t close_file(fuse_fd_t id){
      PDBG("sync cached file called");
      File_meta* fileinfo;
      try{
        fileinfo = _items.at(id);
      }
      catch(std::out_of_range e){
        PERR("close failed");
        return E_FAIL;
      }
      fileinfo->flush_cache();
      return S_OK;
    }

    /* This will write to locked virt-addr*/
    status_t write(fuse_fd_t id, const void * value, size_t size, size_t file_offset = 0 ){
      PDBG("cached write called");

      File_meta* fileinfo = _items.at(id);
      if(fileinfo->_cached_pages.size()>1 || size + file_offset > PAGE_CACHE_SIZE)
        throw General_exception("Need more than one cache page");
      void* buf = (fileinfo->_cached_pages.front())->vaddr;
      memcpy(buf, value, size);
      return S_OK;
    }

    status_t read(fuse_fd_t id, void * value, size_t size, size_t file_offset = 0 ){

      PDBG("cached read called");

      File_meta* fileinfo = _items.at(id);
      if(fileinfo->_cached_pages.size()>1 || size + file_offset > PAGE_CACHE_SIZE)
        throw General_exception("Need more than one cache page");
      void* buf = (fileinfo->_cached_pages.front())->vaddr;
      memcpy(value, buf, size);

      return S_OK;
    }

  private:

      // ownership of this store
      std::string _owner;      
      std::string _name;

      Component::IKVStore *_store;
      pool_t _pool;
      std::map<fuse_fd_t, File_meta *> _items;  
      std::atomic<fuse_fd_t> _assigned_ids; //current assigned ids, to identify each file
};
#endif

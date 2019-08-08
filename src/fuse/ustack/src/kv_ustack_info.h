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

#if 0
/**
 * Private information for this mounting.
 *
 * Allocate the fuse daemon internal fh; interact with kvstore
 * Deprecated, use the cached version below instead.
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
      for(auto item = _files.begin(); item!= _files.end(); item++){
        filenames.push_back(item->second->name);
      }
      return filenames;
    }; 

    fuse_fd_t insert_item(std::string key){
      fuse_fd_t id=  ++_asigned_ids;

      _files.insert(std::make_pair(id, new File_meta(key)));
      return id;
    }

    status_t remove_item(uint64_t id){
      status_t ret;
      std::string key =  _files.at(id)->name;

      ret = _store->erase(_pool, key);
      delete _files.at(id);
      _files.erase(id);
      --_asigned_ids;
      return ret;
    }

    /*
     * look up this file based on the filename
     *
     * @return 0 if not found
     */
    fuse_fd_t get_id(std::string item){
      for(const auto & i : _files){
        if(i.second->name == item)
          return i.first;
      }
      return 0;
    }

    /* get the file size 
     */
    size_t get_item_size(fuse_fd_t id){
      return _files.at(id)->size;
    }

    void set_item_size(fuse_fd_t id, size_t size){
      _files.at(id)->size = size;
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
      const std::string key = _files.at(id)->name;
      assert(value !=  NULL);
      return _store->put(_pool, key, value, size);
    }

    status_t read(fuse_fd_t id, void * value, size_t size, size_t file_offset = 0){
      const std::string key = _files.at(id)->name;
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
      std::map<fuse_fd_t, File_meta *> _files;  

  private:

      std::atomic<fuse_fd_t> _asigned_ids; //current assigned ids, to identify each file

};
#endif


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
      File_meta(std::string name, IKVStore * store, IKVStore::pool_t pool): filename(name), _store(store), _pool(pool){}

      ~File_meta(){}

      /** lock objects as page cache*/
      inline status_t cache_add_entry(size_t nr_entries = 1){
        for(auto i = 0; i < nr_entries ; i++){
          page_cache_entry * entry = new page_cache_entry();
          std::string obj_key = filename + "#seg-" + std::to_string(_nr_cached_pages);
          size_t value_len = PAGE_CACHE_SIZE;

          // If same file is opened again, contents shall be the same
          if(S_OK!= _store->lock(_pool, obj_key, IKVStore::STORE_LOCK_WRITE, entry->vaddr, value_len, entry->locked_key)){
            throw General_exception("lock file cached failed");
            }

          PDBG("%s: filename (%s): \n\t get locked_key[%lu] 0x%lx", __func__, filename.c_str(), _nr_cached_pages, uint64_t(entry->locked_key));
          _cached_pages.push_back(entry);
          _nr_cached_pages += 1;
        }
        return S_OK;
      }

      status_t populate_cache(){
        size_t requested_nr_pages = 1; // for new file use one page 
        if(size){ // a previous file
          requested_nr_pages = (size + PAGE_CACHE_SIZE -1) / PAGE_CACHE_SIZE;
          assert(requested_nr_pages > 0);
        }
        return  cache_add_entry(requested_nr_pages);
      }

      /** Need to expand current file*/
      status_t might_enlarge_file(size_t new_size){
        status_t ret = E_FAIL;
        if(new_size <= size) return S_OK;
        if(new_size <= allocated_space()) {
          size = new_size;
          return S_OK;
        }

        // actually lock more objs
        size_t requested_nr_pages = (new_size + PAGE_CACHE_SIZE -1) / PAGE_CACHE_SIZE - _nr_cached_pages;
        assert(requested_nr_pages > 0);

        try{
        ret =  cache_add_entry(requested_nr_pages);
        }catch(...){
          PERR("failed in lock more objs, %lu + %lu", _nr_cached_pages, requested_nr_pages);
        }
        size = new_size;
        return ret;
      }


      /* TODO: I can only flush one time!*/
      status_t flush_cache(){
        // while(! _cached_pages.empty()){
        int i = 0;
        for(auto it = _cached_pages.begin(); it != _cached_pages.end();){

          auto entry = *it;
          PDBG("%s:filename:(%s): \n\t trying to unlock locked_key[%d] 0x%lx", __func__,  filename.c_str(), i++ ,uint64_t(entry->locked_key));
          _store->unlock(_pool, entry->locked_key);
          _cached_pages.erase(it);
          _nr_cached_pages --;
        }
        assert(_nr_cached_pages == 0);
        PDBG("%s:filename:(%s): \n\t All flushed", __func__, filename.c_str());
        return S_OK;
      }


      std::string filename;
      size_t size = 0;

      size_t _nr_cached_pages = 0;
      std::vector<page_cache_entry *> _cached_pages; /** just an array of locked regions*/
      std::mutex file_mutex;

      inline size_t allocated_space(){
        return _nr_cached_pages* PAGE_CACHE_SIZE;
      }

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
      for(auto item = _files.begin(); item!= _files.end(); item++){
        filenames.push_back(item->second->filename);
      }
      return filenames;
    }; 

    fuse_fd_t insert_item(std::string filename){
      fuse_fd_t id=  ++_assigned_ids;

      assert(_pool);
      _files.insert(std::make_pair(id, new File_meta(filename, _store, _pool)));
      return id;
    }

    status_t remove_item(uint64_t id){
      status_t ret;
      std::string key =  _files.at(id)->filename;

      ret = _store->erase(_pool, key);
      delete _files.at(id);
      _files.erase(id);
      --_assigned_ids;
      return ret;
    }

    /*
     * look up this file based on the filename
     *
     * @return 0 if not found
     */
    fuse_fd_t get_id(std::string item){
      for(const auto & i : _files){
        if(i.second->filename == item)
          return i.first;
      }
      return 0;
    }

    /* get the file size 
     */
    size_t get_item_size(fuse_fd_t id){
      return _files.at(id)->size;
    }

    void set_item_size(fuse_fd_t id, size_t size){
      _files.at(id)->size = size;
    }

    // Currently cannot open before previous close is not persistent
    status_t open_file(fuse_fd_t id)
    {
      File_meta* fileinfo = _files.at(id);
      fileinfo->file_mutex.lock();
      fileinfo->populate_cache();
      return S_OK;
    }


    /* This will unlock the obj, thus writing persist data to pool*/
    status_t close_file(fuse_fd_t id){
      PDBG("sync cached file called");
      File_meta* fileinfo;
      try{
        fileinfo = _files.at(id);
      }
      catch(std::out_of_range e){
        PERR("close failed");
        return E_FAIL;
      }
      fileinfo->flush_cache();

      fileinfo->file_mutex.unlock();
      return S_OK;
    }


    /* This will write to locked virt-addr*/
    status_t write(fuse_fd_t id, const void * value, size_t size, size_t file_offset){
      PDBG("cached write called");

      File_meta* fileinfo = _files.at(id);
      fileinfo->might_enlarge_file(size + file_offset);

      size_t bytes_left = size;
      char * p = (char *)value;
      size_t io_size;

      size_t cur_pageid = file_offset/PAGE_CACHE_SIZE;
      size_t cur_pageoff = file_offset%PAGE_CACHE_SIZE;


      while(bytes_left > 0){
        char * target_addr;
        if(cur_pageoff){
          size_t page_leftover = PAGE_CACHE_SIZE - (file_offset%PAGE_CACHE_SIZE);
          io_size = size < page_leftover? size: page_leftover;
          target_addr =  (char *)((fileinfo->_cached_pages[cur_pageid++])->vaddr) + cur_pageoff;
          cur_pageoff = 0;
        }
        else{
          io_size = bytes_left<PAGE_CACHE_SIZE? bytes_left:PAGE_CACHE_SIZE;
          target_addr =  (char *)((fileinfo->_cached_pages[cur_pageid++])->vaddr);
        }

        memcpy(target_addr, p, io_size);

/*        if(((char*)target_addr)[0] == 0 ){*/
          //PWRN("zero value written!");
        /*}*/

        p += io_size;
        bytes_left -= io_size;
      }

      return S_OK;
    }

    status_t read(fuse_fd_t id, void * value, size_t size, size_t file_offset){

      PDBG("cached read called");

      File_meta* fileinfo = _files.at(id);

      size_t bytes_left = size;
      char * p = (char *)value;
      size_t io_size;

      size_t cur_pageid = file_offset/PAGE_CACHE_SIZE;
      size_t cur_pageoff = file_offset%PAGE_CACHE_SIZE;

      while(bytes_left > 0){
        char * source_addr;
        if(cur_pageoff){
          size_t page_leftover = PAGE_CACHE_SIZE - (file_offset%PAGE_CACHE_SIZE);
          io_size = size < page_leftover? size: page_leftover;
          source_addr =  (char *)((fileinfo->_cached_pages[cur_pageid++])->vaddr) + cur_pageoff;
          cur_pageoff = 0;
        }
        else{
          io_size = bytes_left<PAGE_CACHE_SIZE? bytes_left:PAGE_CACHE_SIZE;
          source_addr =  (char *)((fileinfo->_cached_pages[cur_pageid++])->vaddr);
        }

        //if(((char*)source_addr)[0]== 0){
          //PWRN("zero value read!");
        /*}*/
        memcpy(p, source_addr, io_size);

        p += io_size;
        bytes_left -= io_size;
      }
      return S_OK;
    }

  private:

      // ownership of this store
      std::string _owner;      
      std::string _name;

      Component::IKVStore *_store;
      pool_t _pool;
      std::map<fuse_fd_t, File_meta *> _files;  
      std::atomic<fuse_fd_t> _assigned_ids; //current assigned ids, to identify each file
};
#endif

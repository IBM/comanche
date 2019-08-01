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

#include <api/components.h>
#include <api/kvstore_itf.h>

/*
 * private information for this mounting.
 * Allocate the fuse daemon internal fh; interact with kvstore
 */
class KV_ustack_info{
  public:
    using pool_t     = uint64_t;
    using fuse_fd_t = uint64_t;

    struct File_meta{
      size_t size;
    };

    KV_ustack_info(const std::string owner, const std::string name, Component::IKVStore *store)
      :_owner(owner), _name(name), _store(store), _asigned_ids(0){


      // TODO: why hardcopied size?
      _pool = _store->create_pool("pool-kvfs-simple", MB(12));
    }

    virtual ~KV_ustack_info(){
    }

    Component::IKVStore *get_store(){
      return _store;
    }

    std::unordered_map<fuse_fd_t, std::string> get_all_items(){
      return _items;
    }; 

    /*
     * get a available file id
     */
    fuse_fd_t alloc_id(){
      return ++_asigned_ids;
    }

    uint64_t insert_item(fuse_fd_t id, std::string key){
      _items.insert(std::pair<fuse_fd_t, std::string>(id, key));
    }

    status_t remove_item(uint64_t id){
      status_t ret;

      ret = _store->erase(_pool, _items[id]);
      _items.erase(id);
      return ret;
    }

    /*
     * look up this file based on the filename
     *
     * @return 0 if not found
     */
    fuse_fd_t get_id(std::string item){
      for(const auto & i : _items){
        if(i.second == item)
          return i.first;
      }
      return 0;
    }

    /* get the file size 
     */
    size_t get_item_size(fuse_fd_t id){
      return _file_meta[id].size;
    }

    void set_item_size(fuse_fd_t id, size_t size){
      _file_meta[id].size = size;
    }

    /**
     * kvstore interaction
     */
    virtual status_t open_file(fuse_fd_t) {return S_OK;};
    virtual status_t close_file(fuse_fd_t) {return S_OK;};

    virtual status_t write(fuse_fd_t id, const void * value, size_t size, size_t file_offset = 0 ){
      if(file_offset){
        throw General_exception("kv_ustack_basic doesn't support offset");
      }
      const std::string key = _items[id];
      assert(value !=  NULL);
      return _store->put(_pool, key, value, size);
    }

    virtual status_t read(fuse_fd_t id, void * value, size_t size, size_t file_offset = 0){
      const std::string key = _items[id];
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


  private:
      // ownership of this store
      std::string _owner;      
      std::string _name;

      Component::IKVStore *_store;
      pool_t _pool;

      std::unordered_map<fuse_fd_t, std::string> _items; // file id and key TODO: put the key to the File_meta!
      std::unordered_map<fuse_fd_t, File_meta> _file_meta;  
      std::atomic<fuse_fd_t> _asigned_ids; //current assigned ids, to identify each file
};

struct page_cache{
  size_t offset;
  bool is_dirty;

};

/** Use lock/unlock to provide page cache*/
class KV_ustack_info_cached: public KV_ustack_info{

  public:

    KV_ustack_info_cached(const std::string owner, const std::string name, Component::IKVStore *store): KV_ustack_info(owner, name, store){
  };


    virtual ~KV_ustack_info_cached(){
    }

    virtual status_t open_file(fuse_fd_t) override
    {
      PDBG("cached open called");
      return S_OK;
    }

    virtual status_t close_file(fuse_fd_t) override{
      PDBG("cached close called");
      return S_OK;
    }

    virtual status_t write(fuse_fd_t id, const void * value, size_t size, size_t file_offset = 0 ) override{
      PDBG("cached write called");
      return S_OK;
    }

    virtual status_t read(fuse_fd_t id, const void * value, size_t size, size_t file_offset = 0 ){

      PDBG("cached read called");
      return S_OK;
    }

  private:

};
#endif

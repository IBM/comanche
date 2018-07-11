/*
 * Description:
 *
 * First created: 2018 Jul 11
 * Last modified: 2018 Jul 11
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
 * private information for this mounting, this only modifies metadata
 */
class KV_ustack_info{
using pool_t     = uint64_t;
  public:
    struct File_meta{
      size_t size;
    };

    KV_ustack_info(const std::string owner, const std::string name, Component::IKVStore *store)
      :_owner(owner), _name(name), _store(store), _asigned_ids(0){


      _pool = _store->create_pool("/mnt/pmem0", "pool-kvfs-simple", MB(12));
("");
    }

    ~KV_ustack_info(){
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
      // ownership of this store
      std::string _owner;      
      std::string _name;

      Component::IKVStore *_store;
      pool_t _pool;

      std::unordered_map<uint64_t, std::string> _items; // file id and key TODO: put the key to the File_meta!
      std::unordered_map<uint64_t, File_meta> _file_meta;  
      std::atomic<uint64_t> _asigned_ids; //current assigned ids, to identify each file
};


#endif

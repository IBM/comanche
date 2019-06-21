/*
 * Description:
 *
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 */

#ifndef PERSIST_SESSION_H_
#define PERSIST_SESSION_H_
#include <api/kvstore_itf.h>
#include "block_manager.h"
#include "state_map.h"

using namespace Component;

namespace nvmestore
{
/** Extra info(order) to free block range*/
using block_region_t =
    void*;  // higher 32bit for lba_start, lower 32bit for order

struct obj_info {
  // TODO better padding
  // Block allocation
  int size;  // value size in bytes

  // block allocation
  lba_t          lba_start;
  block_region_t block_region;  // handle to free this block

  // handle for the metastore lock/unlock
  IKVStore::key_t meta_key;

  // key info
  size_t key_len;
  char*  key_data;  // actual char array follows this with ending '\0'
};

struct buffer_t {
  const size_t      _length;
  const io_buffer_t _io_mem;
  void* const
      _start_vaddr;  // it will equal to _io_mem if using allocate_io_buffer

  buffer_t(size_t length, io_buffer_t io_mem, void* start_vaddr)
      : _length(length), _io_mem(io_mem), _start_vaddr(start_vaddr)
  {
  }

  ~buffer_t() {}

  inline size_t length() const { return _length; }
  inline size_t io_mem() const { return _io_mem; }
  inline void*  start_vaddr() const { return _start_vaddr; }
};

class persist_session {
 private:
  using obj_info_t      = struct obj_info;
  using obj_info_pool_t = IKVStore::pool_t;

 public:
  static constexpr bool option_DEBUG = false;
  using lock_type_t                  = IKVStore::lock_type_t;
  using key_t = uint64_t;  // virt_addr is used to identify each obj
  persist_session(IKVStore*        metastore,
                  IKVStore::pool_t obj_info_pool,
                  std::string      path,
                  size_t           io_mem_size,
                  Block_manager*   blk_manager,
                  State_map*       ptr_state_map)
      : _meta_store(metastore), _meta_pool(obj_info_pool), _path(path),
        _io_mem_size(io_mem_size), _blk_manager(blk_manager),
        p_state_map(ptr_state_map), _num_objs(0)
  {
    _io_mem = _blk_manager->allocate_io_buffer(_io_mem_size, 4096,
                                               Component::NUMA_NODE_ANY);
  }

  ~persist_session()
  {
    if (option_DEBUG) PLOG("CLOSING session");
    if (_io_mem) _blk_manager->free_io_buffer(_io_mem);
  }

  std::unordered_map<uint64_t, std::string>& get_locked_regions()
  {
    return _locked_regions;
  }
  std::string get_path() & { return _path; }

  size_t           get_count() { return _num_objs; }
  IKVStore::pool_t get_obj_info_pool() const { return _meta_pool; }

  status_t alloc_new_object(const std::string& key,
                            size_t             value_len,
                            obj_info_t*&       out_blkmeta);

  /** Erase Objects*/
  status_t erase(const std::string& key);

  status_t check_exists(const std::string key) const;

  /** Put and object*/
  status_t put(const std::string& key,
               const void*        valude,
               size_t             value_len,
               unsigned int       flags);

  /** Get an object*/
  status_t get(const std::string& key, void*& out_value, size_t& out_value_len);

  status_t get_direct(const std ::string& key,
                      void*               out_value,
                      size_t&             out_value_len,
                      buffer_t*           memory_handle);

  key_t lock(const std::string& key,
             lock_type_t        type,
             void*&             out_value,
             size_t&            out_value_len);

  status_t unlock(key_t obj_key);

  status_t map(std::function<int(const std::string& key,
                                 const void*        value,
                                 const size_t       value_len)> f);
  status_t map_keys(std::function<int(const std::string& key)> f);

 private:
  // for meta_pmem only

  std::string               _path;
  uint64_t                  _io_mem;      /** dynamic iomem for put/get */
  size_t                    _io_mem_size; /** io memory size */
  nvmestore::Block_manager* _blk_manager;
  State_map*                p_state_map;

  IKVStore*             _meta_store;
  obj_info_pool_t const _meta_pool; /** pool to store hashkey->obj mapping*/
  size_t                _meta_pool_size;

  /** Session locked, io_buffer_t(virt_addr) -> key_str of obj*/
  std::unordered_map<io_buffer_t, std::string> _locked_regions;
  size_t                                       _num_objs;

  status_t may_ajust_io_mem(size_t value_len);
};

}  // namespace nvmestore

#endif

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
class persist_session {
 private:
  using obj_info_t = struct obj_info;

 public:
  static constexpr bool option_DEBUG = false;
  using lock_type_t                  = IKVStore::lock_type_t;
  using key_t = uint64_t;  // virt_addr is used to identify each obj
  persist_session(size_t         pool_size,
                  std::string    path,
                  size_t         io_mem_size,
                  Block_manager* blk_manager,
                  State_map*     ptr_state_map)
      : _path(path), _io_mem_size(io_mem_size), _blk_manager(blk_manager),
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

  std::unordered_map<uint64_t, io_buffer_t>& get_locked_regions()
  {
    return _locked_regions;
  }
  std::string get_path() & { return _path; }

  size_t get_count() { return _num_objs; }

  void alloc_new_object(const std::string& key,
                        size_t             value_len,
                        obj_info_t*&       out_blkmeta)
  {
    throw API_exception("Not implemented");
  }

  /** Erase Objects*/
  status_t erase(const std::string& key)
  {
    throw API_exception("Not implemented");
  }

  /** Put and object*/
  status_t put(const std::string& key,
               const void*        valude,
               size_t             value_len,
               unsigned int       flags)
  {
    throw API_exception("Not implemented");
  }

  /** Get an object*/
  status_t get(const std::string& key, void*& out_value, size_t& out_value_len)
  {
    throw API_exception("Not implemented");
  }

  status_t get_direct(const std ::string& key,
                      void*               out_value,
                      size_t&             out_value_len,
                      buffer_t*           memory_handle)
  {
    throw API_exception("Not implemented");
  }

  key_t lock(const std::string& key,
             lock_type_t        type,
             void*&             out_value,
             size_t&            out_value_len)
  {
    throw API_exception("Not implemented");
  }

  status_t unlock(key_t obj_key) { throw API_exception("Not implemented"); }

  status_t map(std::function<int(const std::string& key,
                                 const void*        value,
                                 const size_t       value_len)> f)
  {
    throw API_exception("Not implemented");
  }
  status_t map_keys(std::function<int(const std::string& key)> f)
  {
    throw API_exception("Not implemented");
  };

 private:
  // for meta_pmem only

  size_t                    _pool_size;
  std::string               _path;
  uint64_t                  _io_mem;      /** dynamic iomem for put/get */
  size_t                    _io_mem_size; /** io memory size */
  nvmestore::Block_manager* _blk_manager;
  State_map*                p_state_map;

  /** Session locked, io_buffer_t(virt_addr) -> pool hashkey of obj*/
  std::unordered_map<io_buffer_t, uint64_t> _locked_regions;
  size_t                                    _num_objs;

  status_t may_ajust_io_mem(size_t value_len);
};

}  // namespace nvmestore

#endif

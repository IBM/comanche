/*
 * dawn_store.hpp
 *
 *  Created on: May 21, 2019
 *      Author: yanzhao
 */

#ifndef INCLUDE_DAWN_STORE_HPP_
#define INCLUDE_DAWN_STORE_HPP_


#include<stdio.h>
#include<stdlib.h>

#include<vector>
#include<iostream>

// Memory Headers
#include <sys/mman.h>


// Comanche Headers
#include<common/exceptions.h>
#include<common/logging.h>
#include<common/types.h>
#include<api/components.h>
#include<api/kvstore_itf.h>

// CUDA Headers
#include<cuda.h>
#include<cuda_runtime_api.h>

// xms related headers
#include "core/physical_memory.h"
#include "core/xms.h"
#include "api/memory_itf.h"

// dawn-app headers
#include "cudautility.hpp"
#include "kvstoreutility.hpp"

 // DAWN options
#define DAWN_TEST // For module testing

using namespace Component;
using namespace std;

class DawnKVStore_wrapper {
private:
  /* Comanche Config */
  string _comanche_component = "libcomanche-dawn-client.so";
  uuid_t _component_id = dawn_client_factory;

  /* DawnKVStore Config */
  unsigned _debug_level = 0;
  string _owner = "public";
  string _addr_port = "10.0.0.51:11900";
  string _device = "mlx5_0";

  IKVStore * _kvstore;

  /* Put large data -> put_direct*/
  IKVStore::memory_handle_t _direct_handle = nullptr;
  size_t _direct_size = 0;
  void * _direct_mem;

  vector<Pool_wrapper*> * _pools_vptr; //

public:
  DawnKVStore_wrapper(string owner, string addr_port, string device, \
      string comanche_component = "libcomanche-dawn-client.so", \
	  uuid_t component_id = dawn_client_factory, unsigned debug_level = 0):
    _owner(owner),
	_addr_port(addr_port),
	_device(device),
    _comanche_component(comanche_component),
    _component_id(component_id),
	_debug_level(debug_level)
  {
    IBase * comp = load_component(_comanche_component.c_str(), _component_id);
    assert(comp);
    IKVStore_factory * fact = (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());
    _kvstore = fact->create(_debug_level,
                          _owner,
                          _addr_port.c_str(),
                          _device.c_str());

    PINF("Create KVStore: %s", _comanche_component.c_str());
    fact->release_ref();
    _pools_vptr = new vector<Pool_wrapper*>();
  }

  DawnKVStore_wrapper(string owner, string addr_port, string device, unsigned debug_level = 0):
	DawnKVStore_wrapper(owner, addr_port, device, \
		  "libcomanche-dawn-client.so", \
		  dawn_client_factory, debug_level) { }
  ~DawnKVStore_wrapper() {
    _kvstore -> release_ref();
  }

  inline status_t put(Pool_wrapper * pool_wrapper, const std::string key, const void * value, const size_t value_len) {
	  //status_t rv =_kvstore->put(pool_wrapper->_pool, key, value, value_len);
	  status_t rv;
	  if (1) {//rv == IKVStore::E_TOO_LARGE) {
		  if (_direct_size == 0 || _direct_size < value_len) {
			  if (_direct_handle != nullptr) {
				  _kvstore->unregister_direct_memory(_direct_handle);
			  }
			  _direct_size = value_len;
			  _direct_mem = aligned_alloc(MiB(2), _direct_size); // TODO: Free
			  madvise(_direct_mem, _direct_size, MADV_HUGEPAGE);
			  _direct_handle = _kvstore->register_direct_memory(_direct_mem, _direct_size);
		  }
		  memcpy(_direct_mem, value, value_len);
		  rv = _kvstore->put_direct(pool_wrapper->_pool, key, value, value_len, _direct_handle);
	  }
	  return rv;
  }

  inline status_t put_direct(Pool_wrapper * pool_wrapper, const std::string key, const void * value, const size_t value_len, IKVStore::memory_handle_t handle) {
      return _kvstore->put_direct(pool_wrapper->_pool, key, value, value_len, handle);
  }

  inline status_t get_direct(Pool_wrapper * pool_wrapper, const std::string key, void * out_value, size_t out_value_len, IKVStore::memory_handle_t handle) {
    return _kvstore->get_direct(pool_wrapper->_pool, key, out_value, out_value_len, handle);
  }

  inline status_t get(Pool_wrapper * pool_wrapper, const std::string key, void * & out_value, size_t& out_value_len) {
    return _kvstore->get(pool_wrapper->_pool, key, out_value, out_value_len);
  }

  inline IKVStore::memory_handle_t register_direct_memory(void * vaddr, size_t len) {
	  return _kvstore->register_direct_memory(vaddr, len);
  }

  inline status_t unregister_direct_memory(IKVStore::memory_handle_t handle) {
	  return _kvstore->unregister_direct_memory(handle);
  }

  bool create_pool(Pool_wrapper * pool_wrapper) {
	  try {
		  pool_wrapper->_pool = _kvstore->create_pool(pool_wrapper->_pool_dir,
		              pool_wrapper->_pool_name,
		              pool_wrapper->_pool_size);
		  PINF("Create Pool id: %d", pool_wrapper->_pool);
		  if (pool_wrapper->_pool == 0 || pool_wrapper->_pool == IKVStore::POOL_ERROR) {
			  PINF("Error: create pool");
			  PINF("Try to open the pool");
			  pool_wrapper->_pool = _kvstore->open_pool(pool_wrapper->_pool_dir, pool_wrapper->_pool_name);
		  }
	  }
	  catch(...) {
		  PINF("Error: create pool error!");
	  }
  }

  bool open_pool(Pool_wrapper * pool_wrapper) {
    // Create a pool. TODO: An array of pools.
    try {
        pool_wrapper->_pool = _kvstore->open_pool(pool_wrapper->_pool_dir,
            pool_wrapper->_pool_name);
    }
    catch(...) {
    	PINF("Error: create pool error!");
    }
    _pools_vptr->push_back(pool_wrapper);
    return (pool_wrapper->_pool != Component::IKVStore::POOL_ERROR);
  }

  void close_pool(Pool_wrapper * pool_wrapper) {
    _kvstore -> close_pool(pool_wrapper->_pool);
  }
};

#ifdef DAWN_STORE_TEST

static void compare_buf(uint32_t *ref_buf, uint32_t *buf, size_t size)
{
    int diff = 0;
    assert(size % 4 == 0U);
    for(unsigned  w = 0; w<size/sizeof(uint32_t); ++w) {
                if (ref_buf[w] != buf[w]) {
                        if (diff < 10)
                                printf("[word %d] %08x != %08x\n", w, buf[w], ref_buf[w]);
                        ++diff;
                }
    }
    if (diff) {
        cout << "check error: diff(s)=" << diff << endl;
    }
}
#endif /* DAWN_STORE_TEST */


#endif /* INCLUDE_DAWN_STORE_HPP_ */

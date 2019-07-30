/*
 * gpu_direct.hpp
 *
 *  Created on: Jun 21, 2018
 *      Author: Yanzhao Wu
 */

#ifndef GPU_DIRECT_HPP_
#define GPU_DIRECT_HPP_

#include<stdio.h>
#include<stdlib.h>

#include<vector>
#include<iostream>
// Comanche Headers
#include<common/exceptions.h>
#include<common/logging.h>
#include<api/components.h>
#include<api/kvstore_itf.h>

// CUDA Headers
#include<cuda.h>
#include<cuda_runtime_api.h>

// GDR Headers (gdrcopy)
#include<gdrapi.h>

// xms related headers
#include "core/physical_memory.h"
#include "core/xms.h"
#include "api/memory_itf.h"

#include "core/dpdk.h"

 // GPU_DIRECT options
#define GPU_DIRECT_TEST // For module testing

#ifndef CUDA_CHECK
#define CUDA_CHECK(x)  if(x != cudaSuccess) \
    PERR("error: cuda err=%s", cudaGetErrorString (cudaGetLastError()));
#endif

#ifndef CU_CHECK
#define CU_CHECK(x)  if(x != CUDA_SUCCESS ) \
    PERR("error: cuda err=%s", cudaGetErrorString (cudaGetLastError()));
#endif

using namespace Component;
using namespace std;

class GDR_wrapper;

class GDR_ptr {
  friend class GDR_wrapper;
  /* Mapping the GPU address to the host memory. */
private:
  CUdeviceptr _d_ptr;
  void * _h_ptr;
  gdr_mh_t _gdr_mh;
  gdr_info_t _gdr_info;
  GDR_wrapper * _gdr; // Pointer back to the GDR_wrapper

public:

  GDR_ptr(CUdeviceptr d_ptr, void * h_ptr, GDR_wrapper * gdr):
    _d_ptr(d_ptr),
    _h_ptr(h_ptr),
    _gdr(gdr) {}

  GDR_ptr():
    GDR_ptr(NULL, NULL, NULL) {}

  CUdeviceptr get_gpu_ptr() {
    return _d_ptr;
  }

  bool set_gpu_ptr(CUdeviceptr d_ptr) {
    this->_d_ptr = d_ptr;
    return (this->_d_ptr == d_ptr);
  }

  void * get_cpu_ptr() {
    return _h_ptr;
  }

  bool set_cpu_ptr(void* h_ptr) {
    this->_h_ptr = h_ptr;
    return (this->_h_ptr == h_ptr);
  }

  GDR_wrapper * get_gdr() {
    return _gdr;
  }

  bool set_gdr_wrapper(GDR_wrapper * gdr) {
    this->_gdr = gdr;
    return (this->_gdr == gdr);
  }
};



class GDR_wrapper{
private:
  /* GDR Copy Config*/
  gdr_t _gdr;
  int _device_id = 0;
  vector<GDR_ptr*>* _gdr_ptrs;

public:
  GDR_wrapper() {
    CUDA_CHECK(cudaSetDevice(_device_id));
    void *dummy;
    CUDA_CHECK(cudaMalloc(&dummy, 0));
    _gdr = gdr_open();
    assert(_gdr);
    _gdr_ptrs = new vector<GDR_ptr*>();
  }
  ~GDR_wrapper() {
    //TODO: Clear gdr pointers in _gdr_ptrs;
    assert(!gdr_close(_gdr));
  }

  inline int pin_buffer(unsigned long adr, size_t size, uint64_t p2p_token, uint32_t va_space, gdr_mh_t *gdr_mh) {
    return gdr_pin_buffer(_gdr, adr, size, p2p_token, va_space, gdr_mh);
  }

  inline int map(gdr_mh_t gdr_mh, void ** va, size_t size) {
    return gdr_map(_gdr, gdr_mh, va, size);
  }

  inline int unmap(gdr_mh_t gdr_mh, void * va, size_t size) {
    return gdr_unmap(_gdr, gdr_mh, va, size);
  }

  inline int unpin_buffer(gdr_mh_t gdr_mh) {
    return gdr_unpin_buffer(_gdr, gdr_mh);
  }

  void map_gdr_gpu_ptr(GDR_ptr * gdr_ptr, size_t buffer_size, CUdeviceptr deviceptr = 0) {
    if(gdr_ptr->_d_ptr == NULL) {
        if(deviceptr == 0) {
	    void * tmp_ptr;
            CUDA_CHECK(cudaMalloc(&tmp_ptr, buffer_size));// TODO: Need free
	    gdr_ptr->_d_ptr = (CUdeviceptr) tmp_ptr;
        } else {
            gdr_ptr->_d_ptr = deviceptr;
        }
    } else {
        if(deviceptr != 0) {
            assert(gdr_ptr->_d_ptr == deviceptr);
        } else {
            assert(gdr_ptr->_d_ptr);
        }
    }

    if(gdr_pin_buffer(_gdr,
                        gdr_ptr->_d_ptr,
                        buffer_size,
                        0,
                        0,
                        &gdr_ptr->_gdr_mh))
        throw General_exception("gdr_pin_buffer failed");

    void *bar_ptr  = NULL;

    if(gdr_map(_gdr, gdr_ptr->_gdr_mh, &bar_ptr, buffer_size), 0)
        throw General_exception("gdr_map failed");
    assert(bar_ptr);

    PLOG("gdr_map returned bar_ptr=%p", bar_ptr);
    if(gdr_get_info(_gdr, gdr_ptr->_gdr_mh, &gdr_ptr->_gdr_info))
            throw General_exception("gdr_get_info failed");
    int off = ((unsigned long)gdr_ptr->_d_ptr) - gdr_ptr->_gdr_info.va;
    gdr_ptr->_h_ptr = (void*) ((char *)bar_ptr + off);
  }

  void map_gdr_gpu_ptr_xms(GDR_ptr * gdr_ptr, size_t buffer_size, void * vaddr_hint, CUdeviceptr deviceptr = 0) {
    //DPDK::eal_init(512);
    Core::Physical_memory memory_instance;
    if(gdr_ptr->_d_ptr == NULL) {
        if(deviceptr == 0) {
	    void * tmp_ptr;
            CUDA_CHECK(cudaMalloc(&tmp_ptr, buffer_size+MB(4)));
	    gdr_ptr->_d_ptr = (CUdeviceptr) tmp_ptr;
        } else {
            gdr_ptr->_d_ptr = deviceptr;
        }
    } else {
        if(deviceptr != 0) {
            assert(gdr_ptr->_d_ptr == deviceptr);
        } else {
            assert(gdr_ptr->_d_ptr);
        }
    }

    if(gdr_pin_buffer(_gdr,
                        gdr_ptr->_d_ptr,
                        buffer_size+MB(4),
                        0,
                        0,
                        &gdr_ptr->_gdr_mh))
        throw General_exception("gdr_pin_buffer failed");

    void *bar_ptr  = NULL;

    if(gdr_map(_gdr, gdr_ptr->_gdr_mh, &bar_ptr, buffer_size+MB(4)), 0)
        throw General_exception("gdr_map failed");
    assert(bar_ptr);

    PLOG("gdr_map returned bar_ptr=%p", bar_ptr);
    if(gdr_get_info(_gdr, gdr_ptr->_gdr_mh, &gdr_ptr->_gdr_info))
        throw General_exception("gdr_get_info failed");
    int off = ((unsigned long)gdr_ptr->_d_ptr) - gdr_ptr->_gdr_info.va;

    void *host_vaddr = (void*) ((char *)bar_ptr + off);
    addr_t host_paddr = xms_get_phys(host_vaddr);
    PLOG("GDR vaddr=%p paddr=%p", (void*)host_vaddr, (void*)host_paddr);

    addr_t new_paddr = round_up(host_paddr, MB(2));
    unsigned offset = new_paddr - host_paddr;
    void * new_vaddr = ((char*)host_vaddr)+offset;

    new_vaddr = xms_mmap(vaddr_hint, new_paddr, buffer_size);

    PLOG("new paddr=0x%lx vaddr=0x%lx", (addr_t) new_paddr, (addr_t)new_vaddr);
    memset(new_vaddr, 0x0, buffer_size);
    Component::io_buffer_t iob = memory_instance.register_memory_for_io(new_vaddr, new_paddr, buffer_size);
    gdr_ptr->_h_ptr = new_vaddr;
    gdr_ptr->_d_ptr += offset;
  }

};

class KVStore_wrapper;

/* Applications may create different pools with different data features.
 * It is easier to manage and optimize the data access with multiple pools.
 **/
class Pool_wrapper {
  friend class KVStore_wrapper; // TODO: Remove friend definition
private:
  /* Pool Config */
  IKVStore::pool_t _pool;
  string _pool_dir = "/mnt/pmem0/";
  string _pool_name = "test0.pool";
  size_t _pool_size = MB(4096);

  KVStore_wrapper * _kvstore;
public:
  Pool_wrapper(string pool_dir, string pool_name, size_t pool_size):
    _pool_dir(pool_dir),
    _pool_name(pool_name),
    _pool_size(pool_size) {}
};

class KVStore_wrapper {
private:
  /* Comanche Config */
  string _comanche_component = "libcomanche-storefile.so";
  uuid_t _component_id = filestore_factory;

  /* KVStore Config */
  string _owner = "public";
  string _name = "name";
  IKVStore * _kvstore;

  vector<Pool_wrapper*> * _pools_vptr; //

  string _port = ""; // PCIe port for nvmestore

public:
  KVStore_wrapper(string owner, string name, \
      string comanche_component, uuid_t component_id, string port = ""):
    _owner(owner),
    _name(name),
    _comanche_component(comanche_component),
    _component_id(component_id),
    _port(port)
     {
    IBase * comp = load_component(_comanche_component.c_str(), _component_id);
    assert(comp);
    IKVStore_factory * fact = (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());
    PINF("Create KVStore: %s", _comanche_component.c_str());
    if(_comanche_component == "libcomanche-storefile.so") {
        _kvstore = fact->create(_owner, _name);
    } else if(_comanche_component == "libcomanche-nvmestore.so") {
        if(_port != "") {
            _kvstore = fact -> create(_owner, _name, _port);
        } else {
            PERR("No port for NVME device!");
            _kvstore = fact -> create(_owner, _name);
        }
    }
    fact->release_ref();

    _pools_vptr = new vector<Pool_wrapper*>();
  }

  KVStore_wrapper():
      KVStore_wrapper("public", "name", "libcomanche-storefile.so", filestore_factory) {}

  ~KVStore_wrapper() {
    _kvstore -> release_ref();
  }

  inline int put(Pool_wrapper * pool_wrapper, const std::string key, const void * value, const size_t value_len) {
    return _kvstore->put(pool_wrapper->_pool, key, value, value_len);
  }

  inline int get_direct(Pool_wrapper * pool_wrapper, const std::string key, void * out_value, size_t out_value_len) {
    return _kvstore->get_direct(pool_wrapper->_pool, key, out_value, out_value_len);
  }

  inline int get(Pool_wrapper * pool_wrapper, const std::string key, void * & out_value, size_t& out_value_len) {
    return _kvstore->get(pool_wrapper->_pool, key, out_value, out_value_len);
  }

  bool open_pool(Pool_wrapper * pool_wrapper) {
    // Create a pool. TODO: An array of pools.
    try {
        pool_wrapper->_pool = _kvstore->create_pool(pool_wrapper->_pool_dir,
            pool_wrapper->_pool_name,
            pool_wrapper->_pool_size);
    }
    catch(...) { //TODO: Specify the exception.
        PINF("Pool %s%s exists, open the pool!", pool_wrapper->_pool_dir.c_str(),
            pool_wrapper->_pool_name.c_str());
        pool_wrapper->_pool = _kvstore->open_pool(pool_wrapper->_pool_dir,
            pool_wrapper->_pool_name);
    }
    _pools_vptr->push_back(pool_wrapper);
    pool_wrapper->_kvstore = this;
    return (pool_wrapper->_pool != NULL);
  }

  void close_pool(Pool_wrapper * pool_wrapper) {
    _kvstore -> close_pool(pool_wrapper->_pool);
  }
};

#ifdef GPU_DIRECT_TEST

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
#endif /* GPU_DIRECT_TEST */

#endif /* GPU_DIRECT_HPP_ */

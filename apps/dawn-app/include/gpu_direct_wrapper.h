/*
 * gpu_direct_wrapper.h
 *
 *  Created on: Jun 21, 2018
 *      Author: Yanzhao Wu
 */

#ifndef GPU_DIRECT_WRAPPER_H_
#define GPU_DIRECT_WRAPPER_H_


#include<stdio.h>
#include<stdlib.h>


// Comanche Headers
#include<common/exceptions.h>
#include<common/logging.h>
#include<api/components.h>
#include<api/kvstore_itf.h>


// CUDA headers
#include<cuda.h>
#include<cuda_runtime_api.h>

// GDR headers (gdrcopy)
#include<gdrapi.h>

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

class GDR_ptr {
public:
  /* Mapping the GPU address to the host memory. */
  CUdeviceptr _d_ptr;
  void * _h_ptr;
};

class GDR_wrapper{
public: //TODO: Hide the internal variables.

  /* GDR Copy Config*/
  gdr_t _gdr;
  int _device_id = 0;
  //GDR_ptr _ptr; // TODO: GDR_wrapper may have multiple pointers.

public:
  GDR_wrapper() {
    CUDA_CHECK(cudaSetDevice(_device_id));
    void *dummy;
    CUDA_CHECK(cudaMalloc(&dummy, 0));
    _gdr = gdr_open();
    assert(_gdr);
  }
  ~GDR_wrapper() {
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

};


class KVStore_wrapper {

	/* Comanche Config */
	string _comanche_component = "libcomanche-storefile.so";
	uuid_t _component_id = filestore_factory;

	/* KVStore Config */
	string _owner = "public";
	string _name = "name";
	IKVStore * _kvstore;

	/* Pool Config */
	IKVStore::pool_t _pool;
	string _pool_dir = "/mnt/pmem0/";
	string _pool_name = "test0.pool";
	size_t _pool_size = MB(4096);

public:
	  KVStore_wrapper(string owner, string name, string pool_dir, string pool_name):
	    _owner(owner),
	    _name(name),
	    _pool_dir(pool_dir),
	    _pool_name(pool_name) {
	    IBase * comp = load_component(_comanche_component.c_str(), _component_id);
	    assert(comp);
	    IKVStore_factory * fact = (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());

	    _kvstore = fact->create(_owner, _name);//, "09:00.0");
	    fact->release_ref();
	    //ifstream infile((_pool_dir+_pool_name).c_str());
	    // Create a pool. TODO: An array of pools.
	    //if(!infile.good()) {
            try{
		_pool = _kvstore->create_pool(_pool_dir, _pool_name, _pool_size);
	    }
	    catch(...) { //TODO: Specify the exception.
		PINF("Pool %s%s exists, open the pool!", _pool_dir.c_str(), _pool_name.c_str());
		_pool = _kvstore->open_pool(_pool_dir, _pool_name);
	    }

	  }

	  KVStore_wrapper():
	      KVStore_wrapper("public", "name", ".", "test0.pool") {}

	  KVStore_wrapper(string pool_dir, string pool_name):
	      KVStore_wrapper("public", "name", pool_dir, pool_name) {}

	  ~KVStore_wrapper() {
	    _kvstore -> close_pool(_pool);
	    _kvstore -> release_ref();
	  }

	  inline int put(const std::string key, const void * value, const size_t value_len) {
		  return _kvstore->put(_pool, key, value, value_len);
	  }

	  inline int get_direct(const std::string key, void * out_value, size_t out_value_len) {
		  return _kvstore->get_direct(_pool, key, out_value, out_value_len);
	  }

	  inline int get(const std::string key, void * & out_value, size_t& out_value_len) {
		  return _kvstore->get(_pool, key, out_value, out_value_len);
	  }
};

/**
 * GPU_direct_wrapper class definition
 */

class GPU_direct_wrapper {

public: //TODO: Hide the internal variables.

	KVStore_wrapper _kvstore;
	GDR_wrapper _gdr;

};



#endif /* GPU_DIRECT_WRAPPER_H_ */

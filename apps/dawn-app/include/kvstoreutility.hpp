/*
 * kvstoreutility.hpp
 *
 *  Created on: May 22, 2019
 *      Author: yanzhao
 */

#ifndef INCLUDE_KVSTOREUTILITY_HPP_
#define INCLUDE_KVSTOREUTILITY_HPP_

#include<stdio.h>
#include<stdlib.h>

#include<vector>
#include<iostream>

// Comanche Headers
#include<common/exceptions.h>
#include<common/logging.h>
#include<common/types.h>
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

using namespace std;

/* Applications may create different pools with different data features.
 * It is easier to manage and optimize the data access with multiple pools.
 **/
class Pool_wrapper {
public:
  /* Pool Config */
  Component::IKVStore::pool_t _pool;
  string _pool_dir = "/mnt/pmem0/";
  string _pool_name = "test0.pool";
  size_t _pool_size = MB(4096);

  Pool_wrapper(string pool_dir, string pool_name, size_t pool_size):
    _pool_dir(pool_dir),
    _pool_name(pool_name),
    _pool_size(pool_size) {}

  Pool_wrapper(string pool_dir, string pool_name):
    Pool_wrapper(pool_dir, pool_name, MB(4096)) {}
};

#endif /* INCLUDE_KVSTOREUTILITY_HPP_ */

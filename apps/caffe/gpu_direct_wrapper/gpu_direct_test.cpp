/*
 * gpu_direct_test.cpp
 *
 *  Created on: Jul 22, 2018
 *      Author: YanzhaoWu
 */
#define CONFIG_DEBUG
#include "gpu_direct.hpp"

int main() {
  Pool_wrapper * test_pool;
  KVStore_wrapper * kvstore;

  test_pool = new Pool_wrapper("./", "pooltest", MB(32));
  kvstore = new KVStore_wrapper(); // Defautl filestore

  kvstore->open_pool(test_pool);
  string key = "key0";
  string value = "Hello World!";
  size_t value_len = (value.length()+1) * sizeof(char);
  kvstore->put(test_pool, key, value.c_str(), value_len);
  void * tmp_value;
  size_t tmp_value_len;
  kvstore->get(test_pool, key, tmp_value, tmp_value_len);
  assert(tmp_value_len == value_len);
  PLOG("Get Key: %s", (char*) tmp_value);

  delete kvstore;

  GDR_ptr * gdr_ptr;
  GDR_wrapper * gdr;
  gdr_ptr = new GDR_ptr();
  gdr = new GDR_wrapper();
  /* Init SPDK */
  kvstore = new KVStore_wrapper("public", "name", "libcomanche-nvmestore.so", Component::nvmestore_factory, "81:00.0");
  //gdr->map_gdr_gpu_ptr(gdr_ptr, MB(2));
  gdr->map_gdr_gpu_ptr_xms(gdr_ptr, MB(2), (void *)0x900000000);
  delete gdr;
  delete kvstore;
}

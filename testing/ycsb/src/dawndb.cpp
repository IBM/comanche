#include "dawndb.h"
#include <api/components.h>
#include <api/kvstore_itf.h>
#include <common/cpu.h>
#include <common/str_utils.h>
#include <core/dpdk.h>
#include <core/task.h>
#include <gtest/gtest.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <string>

using namespace Component;
using namespace Common;
using namespace std;

Component::IKVStore *client;

void init(Properties &props)
{
  Component::IBase *comp = Component::load_component(
      "libcomanche-dawn-client.so", dawn_client_factory);

  IKVStore_factory *fact =
      (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());
  string username = "luna";
  string address  = props.getProperties("address");
  string dev      = props.getProperties("dev");
  client          = fact->create(debug, username, address, dev);
  fact->release_ref();
}

int get(const string &pool, const string &key, char *value, bool direct = false)
{
  int ret = 0;
  /* open or create pool */
  Component::IKVStore::pool_t pool =
      client->open_pool("/mnt/pmem0/dawn", pool, 0);

  if (pool == Component::IKVStore::POOL_ERROR) {
    /* ok, try to create pool instead */
    pool = client->create_pool("/mnt/pmem0/dawn", pool, GB(1));
  }
  if (direct) {
    size_t length = ycsb::Workload::SIZE;
    auto   handle = client->register_direct_memory(buffer, length);
    ret           = client->get_direct(pool, key, value, length, handle);
    client->unregister_direct_memory(handle);
  }
  else {
    void * b   = nullptr;
    size_t len = 0;
    ret        = client->get(pool, key, b, len);
    memcpy(value, b, len);
    client->free_memory(b);
  }

  client->close_pool(pool);
  return ret;
}

int put(const string &pool,
        const string &key,
        const char *  value,
        bool          direct = false)
{
  /* open or create pool */
  Component::IKVStore::pool_t pool =
      client->open_pool("/mnt/pmem0/dawn", pool, 0);

  if (pool == Component::IKVStore::POOL_ERROR) {
    /* ok, try to create pool instead */
    pool = client->create_pool("/mnt/pmem0/dawn", pool, GB(1));
  }

  size_t length = strlen(value);
  if (direct) {
    auto handle = client->register_direct_memory(value, length);
    ret         = client->put_direct(pool, key, value, length, handle);
    client->unregister_direct_memory(handle);
  }
  else {
    ret = client->put(pool, key, value, length);
  }
  client->close_pool(pool);
  return ret;
}

int update(const string &pool,
           const string &key,
           const char *  value,
           bool          direct = false)
{
  return put(pool, key, value, direct);
}
int erase(const string &pool, const string &key)
{
  /* open or create pool */
  Component::IKVStore::pool_t pool =
      client->open_pool("/mnt/pmem0/dawn", p.c_str(), 0);

  if (pool == Component::IKVStore::POOL_ERROR) {
    /* ok, try to create pool instead */
    pool = client->create_pool("/mnt/pmem0/dawn", p.c_str(), GB(1));
  }
  int ret = client->erase(pool, key);
  client->close_pool(pool);
  return ret;
}
int scan(const string &pool,
         const string &key,
         int           count,
         vector < map<string, string> & results)
{
  return -1;
}

void clean()
{
  client->release_ref();
  return 0;
}

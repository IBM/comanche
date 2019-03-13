#include "memcached.h"
#include <assert.h>
#include <string>

using namespace ycsb;
using namespace std;

Memcached::Memcached(Properties &props, unsigned core) { init(props, core); }
Memcached::~Memcached() { clean(); }

void Memcached::init(Properties &props, unsigned core)
{
  string address  = props.getProperty("address");
  size_t mid      = address.find(":");
  string host     = address.substr(0, mid);
  int    port     = stoi(address.substr(mid + 1));
  memc            = memcached_create(NULL);
  servers = memcached_server_list_append(servers, host.c_str(), port, &rc);
  assert(rc == MEMCACHED_SUCCESS);
  rc      = memcached_server_push(memc, servers);
  assert(rc == MEMCACHED_SUCCESS);
}

int Memcached::get(const string &table,
                   const string &key,
                   char *        value,
                   bool          direct)
{
  size_t value_len;
  uint32_t flags;
  char *   retrieved_value;
  retrieved_value =
      memcached_get(memc, key.c_str(), key.length(), &value_len, &flags, &rc);
  if (rc != MEMCACHED_SUCCESS) return -1;
  assert(value_len == strlen(value));
  memcpy(value, retrieved_value, value_len);
  free(retrieved_value);
  return 0;
}

int Memcached::put(const string &table,
                   const string &key,
                   const string &value,
                   bool          direct)
{
  time_t   expiry = 0;
  uint32_t flags  = 0;
  rc = memcached_set(memc, key.c_str(), key.length(), value.c_str(),
                     value.length(), expiry, flags);
  if (rc != MEMCACHED_SUCCESS) return -1;
  return 0;
}

int Memcached::update(const string &table,
                      const string &key,
                      const string &value,
                      bool          direct)
{
  return put(table, key, value, direct);
}

int Memcached::erase(const string &table, const string &key)
{
  rc = memcached_delete(memc, key.c_str(), key.length(), 0);
  if (rc != MEMCACHED_SUCCESS) return -1;
  return 0;
}

int Memcached::scan(const string &                table,
                    const string &                key,
                    int                           count,
                    vector<pair<string, string>> &results)
{
  return -1;
}

void Memcached::clean()
{
  delete (servers);
  delete (memc);
}


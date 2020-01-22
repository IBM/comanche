/*
   Copyright [2017-2019] [IBM Corporation]
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include "dawndb.h"
#include <common/cpu.h>
#include <common/str_utils.h>
#include <core/dpdk.h>
#include <core/task.h>
#include <gtest/gtest.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

using namespace Component;
using namespace Common;
using namespace std;
using namespace ycsb;

/*
Component::IKVStore::pool_t pool;
Stopwatch                   timer;
*/

DawnDB::DawnDB(Properties &props, unsigned core) { init(props, core); }
DawnDB::~DawnDB()
{
  clean();
}

void DawnDB::init(Properties &props, unsigned core)
{
  Component::IBase *comp = Component::load_component(
      "libcomanche-dawn-client.so", dawn_client_factory);

  IDawn_factory *fact =
      (IDawn_factory *) comp->query_interface(IDawn_factory::iid());
  string username = "luna";
  string address  = props.getProperty("address");
  size_t mid      = address.find(":");
  int    port     = stoi(address.substr(mid + 1));
  int    rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  port += rank / 6;

  char hostname[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(hostname, &name_len);
  char *s = strstr(hostname, "bio"); //if bio1
  /*
  if (s == NULL) {
    address.replace(address.begin() + mid + 1, address.end(), to_string(port));
  }
  else {
      port=11914;
      port+=(rank-48)/6;
    address.assign("10.0.1.94:" + to_string(port));
  }
  */
  cout << "host: " + string(hostname) + ", rank: "+ to_string(rank) +", address: " + address << endl;

  string dev      = props.getProperty("dev");
  int    debug    = stoi(props.getProperty("debug_level", "1"));
  client          = fact->dawn_create(debug, username, address, dev);
  fact->release_ref();
  /*
  IDawn::Shard_stats stats;
  client->get_statistics(stats);
  int opstart = stats.op_request_count;
  std::this_thread::sleep_for(std::chrono::seconds(5));
  client->get_statistics(stats);
  double req_per_sec = (stats.op_request_count - opstart) / 5.0;

  pool = client->open_pool("table" + to_string(core), 0);
  */

  if (pool == Component::IKVStore::POOL_ERROR) {
    /* ok, try to create pool instead */
    pool = client->create_pool(poolname, GB(atoi(props.getProperty("poolsize").c_str())));
  }
}

int DawnDB::get(const string &table,
                const string &key,
                char *        value,
                bool          direct)
{
  int ret = 0;
  /* open or create pool */
  /*
  Component::IKVStore::pool_t pool = client->open_pool(table, 0);

  if (pool == Component::IKVStore::POOL_ERROR) {
    pool = client->create_pool(table, GB(1));
  }
  if (direct) {
    size_t length = strlen(value);
    auto   handle = client->register_direct_memory(value, length);
    ret           = client->get_direct(pool, key, value, length, handle);
    client->unregister_direct_memory(handle);
  }
  else {
  */
  void * b   = nullptr;
  size_t len = 0;
  ret        = client->get(pool, key, b, len);
  memcpy(value, b, len);
  client->free_memory(b);

  // client->close_pool(pool);
  return ret;
}

int DawnDB::put(const string &table,
                const string &key,
                const string &value,
                bool          direct)
{
  /*
  Component::IKVStore::pool_t pool = client->open_pool(table, 0);

  if (pool == Component::IKVStore::POOL_ERROR) {
    pool = client->create_pool(table, GB(1));
  }

  if (direct) {
    auto handle = client->register_direct_memory(value, length);
    ret         = client->put_direct(pool, key, value, length, handle);
    client->unregister_direct_memory(handle);
  }
  else {
  */
  int ret = client->put(pool, key, value.c_str(), value.length());
  // cout << "in db:" << timer.get_lap_time_in_seconds() << endl;
  // client->close_pool(pool);
  return ret;
}

int DawnDB::update(const string &table,
                   const string &key,
                   const string &value,
                   bool          direct)
{
  return put(table, key, value, direct);
}

int DawnDB::erase(const string &table, const string &key)
{
  /*
  Component::IKVStore::pool_t pool = client->open_pool(table, 0);

  if (pool == Component::IKVStore::POOL_ERROR) {
    pool = client->create_pool(table, GB(1));
  }
  */
  int ret = client->erase(pool, key);
  client->close_pool(pool);
  return ret;
}

int DawnDB::scan(const string &                table,
                 const string &                key,
                 int                           count,
                 vector<pair<string, string>> &results)
{
  return -1;
}

void DawnDB::clean()
{
  client->close_pool(pool);
  client->delete_pool(poolname);
  client->release_ref();
}

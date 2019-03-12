#include "aerospikedb.h"
#include <aerospike/aerospike_key.h>
#include <aerospike/as_key.h>
#include <aerospike/as_record.h>
#include <iostream>

using namespace ycsb;
using namespace std;

string AerospikeDB::BIN = "bin";

AerospikeDB::AerospikeDB(Properties &props, unsigned core)
{
  init(props, core);
}
AerospikeDB::~AerospikeDB() { clean(); }

void AerospikeDB::init(Properties &props, unsigned core)
{
  string address = props.getProperty("address");
  size_t mid     = address.find(":");
  string host    = address.substr(0, mid);
  int    port    = stoi(address.substr(mid + 1));
  as_config_init(&config);
  as_config_add_host(&config, host.c_str(), port);
  aerospike_init(&as, &config);
  aerospike_connect(&as, &err);
  assert(err.code == AEROSPIKE_OK);
}

int AerospikeDB::get(const string &table,
                     const string &key,
                     char *        value,
                     bool          direct)
{
  as_key a_key;
  as_record *a_rec = NULL;
  as_key_init(&a_key, "ycsb", table.c_str(), key.c_str());
  aerospike_key_get(&as, &err, NULL, &a_key, &a_rec);
  if (err.code != AEROSPIKE_OK) {
    free(a_rec);
    return -1;
  }
  char *rec = as_record_get_str(a_rec, BIN.c_str());
  memcpy(value, rec, strlen(value));
  free(rec);
  free(a_rec);
  return 0;
}

int AerospikeDB::put(const string &table,
                     const string &key,
                     const string &value,
                     bool          direct)
{
  as_key a_key;
  as_key_init(&a_key, "ycsb", table.c_str(), key.c_str());
  as_record a_rec;
  as_record_init(&a_rec, 1);
  as_record_set_str(&a_rec, BIN.c_str(), value.c_str());
  aerospike_key_put(&as, &err, NULL, &a_key, &a_rec);
  if (err.code != AEROSPIKE_OK) return -1;
  return 0;
}

int AerospikeDB::update(const string &table,
                        const string &key,
                        const string &value,
                        bool          direct)
{
  return put(table, key, value, direct);
}

int AerospikeDB::erase(const string &table, const string &key)
{
  as_key a_key;
  as_key_init(&a_key, "ycsb", table.c_str(), key.c_str());
  aerospike_key_remove(&as, &err, NULL, &a_key);
  if (err.code != AEROSPIKE_OK) return -1;
  return 0;
}

int AerospikeDB::scan(const string &                table,
                      const string &                key,
                      int                           count,
                      vector<pair<string, string>> &results)
{
  return -1;
}

void AerospikeDB::clean()
{
  aerospike_close(&as, &err);
  if (err.code != AEROSPIKE_OK) {
    cout << err.message << " at file: " << err.file << ", line: " << err.line
         << endl;
  }
  aerospike_destroy(&as);
}


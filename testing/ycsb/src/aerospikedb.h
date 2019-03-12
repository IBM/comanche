#ifndef __YCSB_AEROSPIKEDB_H__
#define __YCSB_AEROSPIKEDB_H__

#include <aerospike/aerospike.h>
#include <aerospike/as_config.h>
#include <aerospike/as_error.h>
#include "db.h"
#include "properties.h"

using namespace ycsbutils;

namespace ycsb
{
class AerospikeDB : public DB {
 public:
  AerospikeDB(Properties &props, unsigned core = 0);
  virtual ~AerospikeDB();
  virtual int  get(const string &table,
                   const string &key,
                   char *        value,
                   bool          direct = false) override;
  virtual int  put(const string &table,
                   const string &key,
                   const string &value,
                   bool          direct = false) override;
  virtual int  update(const string &table,
                      const string &key,
                      const string &value,
                      bool          direct = false) override;
  virtual int  erase(const string &table, const string &key) override;
  virtual int  scan(const string &                table,
                    const string &                key,
                    int                           count,
                    vector<pair<string, string>> &results) override;
  virtual void init(Properties &props, unsigned core = 0) override;
  virtual void clean() override;

 private:
  as_error  err;
  as_config config;
  aerospike as;
  static string BIN;
};

}  // namespace ycsb

#endif

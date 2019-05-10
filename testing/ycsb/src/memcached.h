#ifndef __YCSB_MEMCACHED_H__
#define __YCSB_MEMCACHED_H__

//#include <mem_config.h>
#include <libmemcached/memcached.h>
#include "db.h"
#include "properties.h"

using namespace ycsbutils;

namespace ycsb
{
class Memcached : public DB {
 public:
  Memcached(Properties &props, unsigned core = 0);
  virtual ~Memcached();

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
  memcached_server_st *servers = NULL;
  memcached_st *       memc;
  memcached_return     rc;
};

}  // namespace ycsb

#endif

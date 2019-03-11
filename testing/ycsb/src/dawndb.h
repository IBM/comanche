#ifndef __YCSB_DAWNDB_H__
#define __YCSB_DAWNDB_H__

#include <api/components.h>
#include <api/kvstore_itf.h>
#include "../../kvstore/stopwatch.h"
#include "db.h"
#include "properties.h"

using namespace ycsbutils;

namespace ycsb
{
class DawnDB : public DB {
  public:
   DawnDB(Properties &props, unsigned core = 0);
   virtual ~DawnDB();
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
   Component::IKVStore *       client;
   Component::IKVStore::pool_t pool;
};

}  // namespace ycsb

#endif

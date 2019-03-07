#ifndef __YCSB_DAWNDB_H__
#define __YCSB_DAWNDB_H__

#include "db.h"
#include "properties.h"

using namespace ycsbutils;

namespace ycsb
{
class DawnDB : public DB {
  public:
   DawnDB(Properties &props);
   virtual ~DawnDB();
   virtual int  get(const string &table,
                    const string &key,
                    char *        value,
                    bool          direct = false) override;
   virtual int  put(const string &table,
                    const string &key,
                    const char *  value,
                    bool          direct = false) override;
   virtual int  update(const string &table,
                       const string &key,
                       const char *  value,
                       bool          direct = false) override;
   virtual int  erase(const string &table, const string &key) override;
   virtual int  scan(const string &                table,
                     const string &                key,
                     int                           count,
                     vector<pair<string, string>> &results) override;
   virtual void init(Properties &props) override;
   virtual void clean() override;
};

}  // namespace ycsb

#endif

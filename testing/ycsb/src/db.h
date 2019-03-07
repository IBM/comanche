#ifndef __YCSB_DB_H__
#define __YCSB_DB_H__

#include <map>
#include <vector>
#include "properties.h"

using namespace ycsbutils;
using namespace std;

namespace ycsb
{
class DB {
 public:
  virtual int  get(const string &table,
                   const string &key,
                   char *        value,
                   bool          direct = false)                              = 0;
  virtual int  put(const string &table,
                   const string &key,
                   const char *  value,
                   bool          direct = false)                              = 0;
  virtual int  update(const string &table,
                      const string &key,
                      const char *  value,
                      bool          direct = false)                           = 0;
  virtual int  erase(const string &table, const string &key)         = 0;
  virtual int  scan(const string &                table,
                    const string &                key,
                    int                           count,
                    vector<pair<string, string>> &results)           = 0;
  virtual void init(Properties &props)                               = 0;
  virtual void clean()                                               = 0;
  virtual ~DB(){};
};
}  // namespace ycsb

#endif

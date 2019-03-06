#ifndef __YCSB_DAWNDB_H__
#define __YCSB_DAWNDB_H__

#include "db.h"
#include "properties.h"

using namespace ycsbutils;

namespace ycsb
{
class DawnDB::DB {
 public:
  virtual int get(const string &pool,
                  const string &key,
                  char *        value,
                  bool          direct = false) override;
  virtual int put(const string &pool,
                  const string &key,
                  const char *  value,
                  bool          direct = false) override;
  virtual int update(const string &pool,
                     const string &key,
                     const char *  value,
                     bool          direct = false) override;
  virtual int erase(const string &pool, const string &key) override;
  virtual int scan(const string &pool,
                   const string &key,
                   int           count,
                   vector < map<string, string> & results) override;

 private:
  virtual void init(Properties &props) override;
  virtual void clean() override;
};

}  // namespace ycsb

#endif

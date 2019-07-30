#ifndef __YCSB_REDISC_H__
#define __YCSB_REDISC_H__

#include <hiredis/hiredis.h>
#include "db.h"
#include "properties.h"

using namespace ycsbutils;

namespace ycsb
{
class RedisC : public DB {
 public:
  RedisC(Properties & props, unsigned core = 0);
  virtual ~RedisC();
  virtual int  get(const string &table, const string &key, char *value,
                   bool direct = false) override;
  virtual int  put(const string &table, const string &key, const string &value,
                   bool direct = false) override;
  virtual int  update(const string &table, const string &key,
                      const string &value, bool direct = false) override;
  virtual int  erase(const string &table, const string &key) override;
  virtual int  scan(const string &table, const string &key, int count,
                    vector<pair<string, string>> &results) override;
  virtual void init(Properties & props, unsigned core = 0) override;
  virtual void clean() override;

 private:
  redisContext *c = NULL;
  redisReply *  reply;
};
}

#endif

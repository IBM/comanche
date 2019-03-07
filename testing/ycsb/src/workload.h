#ifndef __YCSB_WL_H__
#define __YCSB_WL_H__

#include <vector>
#include "db.h"
#include "generator.h"
#include "properties.h"

using namespace std;

namespace ycsb
{
enum Operation { INSERT, READ, UPDATE, SCAN, READMODIFYWRITE };

class Workload {
 public:
  const int SIZE = 64;
  Workload(Properties& props, DB* db);
  void load();
  void run();
  ~Workload();

 private:
  Properties&                  props;
  DB *       db;
  vector<pair<string, string>> kvs;
  //  ycsbc::Generator<Value>      gen;
  int                          records;
  int                          operations;
  inline string                buildKeyName(uint64_t key_num);
  inline string                buildValue(uint64_t size);
};

}  // namespace ycsb

#endif

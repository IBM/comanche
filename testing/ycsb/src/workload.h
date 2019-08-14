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
#ifndef __YCSB_WL_H__
#define __YCSB_WL_H__

#include <core/task.h>
#include <mutex>
#include <vector>
#include "../../kvstore/statistics.h"
#include "../../kvstore/stopwatch.h"
#include "db.h"
#include "discrete_generator.h"
#include "generator.h"
#include "properties.h"

using namespace std;

namespace ycsb
{
enum Operation { INSERT, READ, UPDATE, SCAN, READMODIFYWRITE };

class Workload {
 public:
  static const int SIZE;
  const string     TABLE;
  Workload(Properties& props, int n, int id);
  void load(double sec);
  void run();
  virtual ~Workload();
  virtual void initialize();
  virtual bool do_work();
  virtual void cleanup();
  void         summarize();

 private:
  Properties&                         props;
  DB*                                 db;
  vector<pair<string, string>>        kvs;
  ycsbc::DiscreteGenerator<Operation> op;
  ycsbc::Generator<uint64_t>*         gen;
  static std::mutex                   _iops_lock;
  static unsigned long                _iops;
  static std::mutex                   _iops_load_lock;
  static unsigned long                _iops_load;
  Stopwatch                           rd;
  Stopwatch                           wr;
  Stopwatch                           up;
  unsigned long                       rd_cnt;
  unsigned long                       wr_cnt;
  unsigned long                       up_cnt;
  RunningStatistics                   rd_stat;
  RunningStatistics                   wr_stat;
  RunningStatistics                   up_stat;
  int                                 n;
  int                                 id;

  int           records;
  int           operations;
  inline string buildKeyName(uint64_t key_num);
  inline string buildValue(uint64_t size);
  void          doRead();
  void          doInsert();
  void          doUpdate();
  void          doScan();
  bool          isready = false;
};

}  // namespace ycsb

#endif

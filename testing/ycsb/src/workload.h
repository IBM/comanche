#ifndef __YCSB_WL_H__
#define __YCSB_WL_H__

#include <core/task.h>
#include <mutex>
#include <vector>
#include "../../kvstore/statistics.h"
#include "../../kvstore/stopwatch.h"
#include "args.h"
#include "db.h"
#include "discrete_generator.h"
#include "generator.h"
#include "properties.h"

using namespace std;

namespace ycsb
{
enum Operation { INSERT, READ, UPDATE, SCAN, READMODIFYWRITE };

class Workload : public Core::Tasklet {
 public:
  static const int    SIZE;
  const string        TABLE;
  Workload(Args& args);
  void load();
  void run();
  virtual ~Workload();
  virtual void initialize(unsigned core) override;
  virtual bool do_work(
      unsigned core) override; /*< called in tight loop; return false to exit */
  virtual void cleanup(unsigned core) override; /*< called once */
  virtual bool ready() override;
  void         summarize();

 private:
  Properties&                  props;
  DB *       db;
  vector<pair<string, string>> kvs;
  ycsbc::DiscreteGenerator<Operation> op;
  ycsbc::Generator<uint64_t>*  gen;
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

  int                          records;
  int                          operations;
  inline string                buildKeyName(uint64_t key_num);
  inline string                buildValue(uint64_t size);
  void                         doRead();
  void                         doInsert();
  void                         doUpdate();
  void                         doScan();
  static bool                  isready;
};

}  // namespace ycsb

#endif

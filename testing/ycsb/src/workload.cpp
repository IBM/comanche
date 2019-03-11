#include "workload.h"
#include <assert.h>
#include <iostream>
#include <string>
#include "../../kvstore/stopwatch.h"
#include "counter_generator.h"
#include "db_fact.h"
#include "generator.h"
#include "properties.h"
#include "uniform_generator.h"
#include "zipfian_generator.h"

using namespace std;
using namespace ycsb;
using namespace ycsbc;

const int    Workload::SIZE  = 32;
unsigned long Workload::_iops      = 0;
unsigned long Workload::_iops_load = 0;
mutex         Workload::_iops_lock;
mutex         Workload::_iops_load_lock;
// DB*           Workload::db;

Workload::Workload(Properties& props) : props(props)
{
}

void Workload::initialize(unsigned core)
{
  db = ycsb::DBFactory::create(props, core);
  assert(db);
  string TABLE = "table" + to_string(core);
  records = stoi(props.getProperty("recordcount"));
  operations = stoi(props.getProperty("operationcount"));
  Generator<uint64_t>* loadkeygen = new CounterGenerator(0);
  for (int i = 0; i < records; i++) {
    pair<string, string> kv(Workload::buildKeyName(loadkeygen->Next()),
                            Workload::buildValue(Workload::SIZE));
    kvs.push_back(kv);
  }
  delete loadkeygen;

  double readproportion   = stod(props.getProperty("readproportion", "0"));
  double updateproportion = stod(props.getProperty("updateproportion", "0"));
  double scanproportion   = stod(props.getProperty("scanproportion", "0"));
  double insertproportion = stod(props.getProperty("insertproportion", "0"));
  if (readproportion > 0) op.AddValue(READ, readproportion);
  if (updateproportion > 0) op.AddValue(UPDATE, updateproportion);
  if (scanproportion > 0) op.AddValue(SCAN, scanproportion);
  if (insertproportion > 0) op.AddValue(INSERT, insertproportion);

  if (props.getProperty("requestdistribution") == "uniform") {
    gen = new UniformGenerator(0, records - 1);
  }
  else if (props.getProperty("requestdistribution") == "zipfian") {
    gen = new ZipfianGenerator(records);
  }
  assert(gen);
  rd.reset();
  wr.reset();
  up.reset();
  rd_cnt = 0;
  wr_cnt = 0;
  up_cnt = 0;
  isready    = true;
}

bool Workload::ready() { return isready; }

bool Workload::do_work(unsigned core)
{
  load();
  if (props.getProperty("run", "0") == "1") {
    run();
  }
  return false;
}

void Workload::load()
{
  int ret;
  wr.reset();
  for (int i = 0; i < records; i++) {
    pair<string, string>& kv = kvs[i];
    // cout << "insert" << endl;
    wr.start();
    ret = db->put(Workload::TABLE, kv.first, kv.second);
    if (ret != 0) {
      throw "Insertion failed in loading phase";
      exit(-1);
    }
    wr.stop();
    double elapse = wr.get_lap_time_in_seconds();
    // cout << timer.get_lap_time_in_seconds() << endl;
    wr_stat.add_value(elapse);
    wr_cnt++;
  }
}

void Workload::run()
{
  rd.reset();
  up.reset();
  for (int i = 0; i < operations; i++) {
    Operation operation = op.Next();
    switch (operation) {
      case READ:
        doRead();
        break;
      case UPDATE:
        doUpdate();
        break;
      case INSERT:
        doInsert();
        break;
      case SCAN:
        doScan();
        break;
    }
  }
}

void Workload::doRead()
{
  // cout << "read" << endl;
  int index = gen->Next();
  if (index > records - 1) {
    throw "Key index overflow!";
    exit(-1);
  }
  string& key = kvs[index].first;
  char   value[Workload::SIZE];
  rd.start();
  int    ret = db->get(Workload::TABLE, key, value);
  if (ret != 0) {
    throw "Read fail!";
    exit(-1);
  }
  rd.stop();
  rd_cnt++;
  double elapse = rd.get_lap_time_in_seconds();
  rd_stat.add_value(elapse);
}

void Workload::doUpdate()
{
  // cout << "update" << endl;
  int index = gen->Next();
  if (index > records - 1) {
    throw "Key index overflow!";
    exit(-1);
  }
  string& key   = kvs[index].first;
  string value = buildValue(Workload::SIZE);
  up.start();
  int    ret   = db->update(Workload::TABLE, key, value.c_str());
  if (ret != 0) {
    throw "Update fail!";
    exit(-1);
  }
  up.stop();
  up_cnt++;
  double elapse = up.get_lap_time_in_seconds();
  up_stat.add_value(elapse);
}

void Workload::doInsert() {}
void Workload::doScan() {}

void Workload::cleanup(unsigned core)
{
  cout << "do clean" << endl;
  rd.stop();
  wr.stop();
  up.stop();

  if (rd_cnt > 0 || up_cnt > 0) {
    cout << "read: " << rd_cnt << ", update: " << up_cnt << endl;
    cout << "Time: " << rd.get_time_in_seconds() + up.get_time_in_seconds()
         << endl;
    unsigned long iops = (rd_cnt + up_cnt) /
                         (rd.get_time_in_seconds() + up.get_time_in_seconds());
    std::lock_guard<std::mutex> g(_iops_lock);
    _iops += iops;
  }
  {
    std::lock_guard<std::mutex> g(_iops_load_lock);
    _iops_load += wr_cnt / wr.get_time_in_seconds();
  }
}

void Workload::summarize()
{
  cout << "[Overall], Load Throughput (ops/sec), " << _iops_load << endl;
  cout << "[Overall], Operation Throughput (ops/sec), " << _iops << endl;
}

Workload::~Workload()
{
  kvs.clear();
  delete gen;
  delete db;
}

inline string Workload::buildKeyName(uint64_t key_num)
{
  key_num = ycsbutils::Hash(key_num);
  return std::string("user").append(std::to_string(key_num));
}

inline string Workload::buildValue(uint64_t size)
{
  return std::string().append(size, RandomPrintChar());
}


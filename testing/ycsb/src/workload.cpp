#include "workload.h"
#include <assert.h>
#include <iostream>
#include <string>
#include "../../kvstore/stopwatch.h"
#include "counter_generator.h"
#include "generator.h"
#include "properties.h"
#include "uniform_generator.h"
#include "zipfian_generator.h"

using namespace std;
using namespace ycsb;
using namespace ycsbc;

const int    Workload::SIZE  = 32;
const string Workload::TABLE = "table";
Stopwatch    timer;

Workload::Workload(Properties& props, DB* db) : props(props), db(db)
{
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
}

void Workload::load()
{
  int ret;
  int op_count = 0;
  timer.reset();
  timer.start();
  for (int i = 0; i < records; i++) {
    pair<string, string>& kv = kvs[i];
    // cout << "insert" << endl;
    ret = db->put(Workload::TABLE, kv.first, kv.second);
    if (ret != 0) {
      throw "Insertion failed in loading phase";
      exit(-1);
    }
    else {
      op_count++;
    }
  }
  timer.stop();
  double elapse = timer.get_lap_time_in_seconds();
  cout << elapse << endl;
  cout << "Throughput for load is: " << op_count / elapse << endl;
  cout << "Latency for load is: " << elapse / op_count * 1000000 << " usec"
       << endl;
}

void Workload::run()
{
  int op_count = 0;
  timer.reset();
  timer.start();
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
    op_count++;
  }
  timer.stop();
  double elapse = timer.get_lap_time_in_seconds();
  cout << elapse << endl;
  cout << "Throughput for run is: " << op_count / elapse << endl;
  cout << "Latency for run is: " << elapse / op_count * 1000000 << " usec"
       << endl;
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
  int    ret = db->get(Workload::TABLE, key, value);
  if (ret != 0) {
    throw "Read fail!";
    exit(-1);
  }
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
  int    ret   = db->update(Workload::TABLE, key, value.c_str());
  if (ret != 0) {
    throw "Update fail!";
    exit(-1);
  }
}

void Workload::doInsert() {}
void Workload::doScan() {}

Workload::~Workload()
{
  kvs.clear();
  delete gen;
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


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
#include "workload.h"
#include <assert.h>
#include <mpi.h>
#include <chrono>
#include <iostream>
#include <string>
#include <time.h>
#include <thread>
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

const int     Workload::SIZE       = 64;
unsigned long Workload::_iops      = 0;
unsigned long Workload::_iops_load = 0;
mutex         Workload::_iops_lock;
mutex         Workload::_iops_load_lock;
// DB*           Workload::db;

Workload::Workload(Properties& props) : props(props) { initialize(); }

void Workload::initialize()
{
  int core;
  MPI_Comm_rank(MPI_COMM_WORLD, &core);
  Generator<uint64_t>* loadkeygen = new CounterGenerator(0);
  db = ycsb::DBFactory::create(props, core);
  assert(db);
  string TABLE = "table" + to_string(core);
  records = stoi(props.getProperty("recordcount"));
  operations = stoi(props.getProperty("operationcount"));
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
}

bool Workload::do_work()
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
  int req_per_sec = stoi(props.getProperty("request"));
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  req_per_sec/=size;
  double sec=(double)records/req_per_sec;

  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < records; i++) {
    pair<string, string>& kv = kvs[i];
    // cout << "insert" << endl;
    wr.start();
    ret = db->put(Workload::TABLE, kv.first, kv.second);
    //  std::this_thread::sleep_for(std::chrono::seconds(1));
    //ret = db->put(Workload::TABLE, "abc", "edf");
    if (ret != 0) {
      throw "Insertion failed in loading phase";
      exit(-1);
    }
    wr.stop();
    double elapse = wr.get_lap_time_in_seconds();
    props.log(to_string(elapse));
    if (elapse < sec) {
      struct timespec ts;
      ts.tv_sec=0;
      ts.tv_nsec=(int)((sec-elapse)*1000000000);
      nanosleep(&ts, NULL);
    }
    //    if (elapse * 1000000 >= 900) wr_stat.add_value(elapse);
    wr_cnt++;
  }
}

void Workload::run()
{
  rd.reset();
  up.reset();
  MPI_Barrier(MPI_COMM_WORLD);
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
  if (elapse * 1000000 >= 900) rd_stat.add_value(elapse);
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
  if (elapse * 1000000 >= 900) up_stat.add_value(elapse);
}

void Workload::doInsert() {}
void Workload::doScan() {}

void Workload::cleanup()
{
  rd.stop();
  wr.stop();
  up.stop();
  MPI_Barrier(MPI_COMM_WORLD);
  cout << "do clean" << endl;

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
  summarize();
}

void Workload::summarize()
{
  unsigned long global_iops_load = 0;
  unsigned long global_iops      = 0;
  unsigned long   local_lat        = 0;
  unsigned long   global_lat       = 0;
  int             rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Reduce(&_iops_load, &global_iops_load, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&_iops, &global_iops, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
             MPI_COMM_WORLD);
  if (rd_cnt > 0) {
    local_lat = rd_stat.getCount();
    MPI_Reduce(&local_lat, &global_lat, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
               MPI_COMM_WORLD);
    if (rank == 0) {
      double p = global_lat / (rd_cnt * size);
      cout << "[Overall], Read percentage of more than 900 us: , " << p << endl;
    }
  }

  if (wr_cnt > 0) {
    local_lat = wr_stat.getCount();
    MPI_Reduce(&local_lat, &global_lat, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
               MPI_COMM_WORLD);
    if (rank == 0) {
      double p = global_lat / (wr_cnt * size);
      cout << "[Overall], Write percentage of more than 900 us: , " << p
           << endl;
    }
  }

  if (up_cnt > 0) {
    local_lat = up_stat.getCount();
    MPI_Reduce(&local_lat, &global_lat, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0,
               MPI_COMM_WORLD);
    if (rank == 0) {
      double p = global_lat / (up_cnt * size);
      cout << "[Overall], Update percentage of more than 900 us: , " << p
           << endl;
    }
  }

  if (rank == 0) {
    cout << "[Overall], Load Throughput (ops/sec), " << global_iops_load << endl;
    cout << "[Overall], Operation Throughput (ops/sec), " << global_iops << endl;
  }
}

Workload::~Workload()
{
    cleanup();
  kvs.clear();
  delete gen;
  delete db;
}

inline string Workload::buildKeyName(uint64_t key_num)
{
  key_num = ycsbutils::Hash(key_num);
  // return std::string("user").append(std::to_string(key_num));
  return (Workload::TABLE + std::to_string(key_num)).substr(0, 16);
}

inline string Workload::buildValue(uint64_t size)
{
  return std::string().append(size, RandomPrintChar());
}


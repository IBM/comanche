#include "workload.h"
#include <iostream>
#include <string>
#include "counter_generator.h"
#include "generator.h"
#include "properties.h"

using namespace std;
using namespace ycsb;
using namespace ycsbc;

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
}

void Workload::load()
{
  int ret;
  for (int i = 0; i < records; i++) {
    pair<string, string> kv = kvs[i];
    cout << "Inserting key: " << kv.first << endl;
    ret = db->put("table", kv.first, kv.second.c_str());
    if (ret != 0) {
      throw "Insertion failed in loading phase";
      exit(-1);
    }
  }
}

void Workload::run() {}

Workload::~Workload() { kvs.clear(); }

inline string Workload::buildKeyName(uint64_t key_num)
{
  key_num = ycsbutils::Hash(key_num);
  return std::string("user").append(std::to_string(key_num));
}

inline string Workload::buildValue(uint64_t size)
{
  return std::string().append(size, RandomPrintChar());
}


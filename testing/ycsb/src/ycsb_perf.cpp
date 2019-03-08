#include "ycsb_perf.h"
#include <stdlib.h>
#include <iostream>
#include "../../kvstore/stopwatch.h"
#include "db.h"
#include "db_fact.h"
#include "workload.h"

using namespace std;

int main(int argc, char * argv[])
{
    if(argc<4)
        show_program_options();

    string operation=argv[1];
    string filename=argv[3];

  ifstream input(filename);

  try {
    props.load(input);
  }
  catch (const string &msg) {
    cerr << msg << endl;
    input.close();
    return -1;
  }

  input.close();
  ycsb::DB *db = ycsb::DBFactory::create(props);
  assert(db);
  ycsb::Workload* wl = new ycsb::Workload(props, db);

  // cpu_mask_t cpus;

  wl->load();
  if (operation == "run") wl->run();
  return 0;
}


void show_program_options()
{
    cout<<"./ycsb_perf load/run -P <workload file>"<<endl;
    exit(-1);
}

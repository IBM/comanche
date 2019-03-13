#include "ycsb_perf.h"
#include <stdlib.h>
#include <iostream>
#include "../../kvstore/get_cpu_mask_from_string.h"
#include "../../kvstore/stopwatch.h"
#include "db_fact.h"
#include "workload.h"

using namespace std;

int main(int argc, char * argv[])
{
    if(argc<4)
        show_program_options();

    string operation = argv[1];
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

    cpu_mask_t cpus;

    try {
      cpus =
          get_cpu_mask_from_string(props.getProperty("cores", "0"));
    }
    catch (...) {
      PERR("%s", "couldn't create CPU mask. Exiting.");
      return 1;
    }

    if (operation == "run") props.setProperty("run", "1");

    Core::Per_core_tasking<ycsb::Workload, Properties &> exp(cpus, props);
    exp.wait_for_all();
    auto first_exp = exp.tasklet(cpus.first_core());
    first_exp->summarize();

    return 0;
}


void show_program_options()
{
    cout<<"./ycsb_perf load/run -P <workload file>"<<endl;
    exit(-1);
}

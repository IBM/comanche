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
#include "ycsb_perf.h"
#include <mpi.h>
#include <stdlib.h>
#include <iostream>
#include "../../kvstore/get_cpu_mask_from_string.h"
#include "../../kvstore/stopwatch.h"
#include "db_fact.h"
#include "workload.h"
#include <string>

using namespace std;
using namespace ycsb;

int main(int argc, char * argv[])
{
  MPI_Init(NULL, NULL);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (argc < 4) show_program_options();

  string operation = argv[1];
  string filename  = argv[3];

  ifstream input(filename);
  props.logfile.open("/tmp/latency-sla",
                     std::ofstream::out | std::ofstream::app);

  try {
    props.load(input);
    }
    catch (const string &msg) {
      cerr << msg << endl;
      input.close();
      return -1;
    }

    input.close();

    if (operation == "run") props.setProperty("run", "1");

    Workload *wl = new Workload(props);
    wl->do_work();
    delete wl;

    props.logfile.close();
    MPI_Finalize();

    return 0;
}


void show_program_options()
{
    cout<<"./ycsb_perf load/run -P <workload file>"<<endl;
    exit(-1);
}

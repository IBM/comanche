/* note: we do not include component source, only the API definition */
#include <common/utils.h>
#include <common/str_utils.h>
#include <core/task.h>
#include <api/components.h>
#include <api/kvstore_itf.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <iostream>

#undef PROFILE

#ifdef PROFILE
#include <gperftools/profiler.h>
#endif

#include "exp_get.h"
#include "exp_get_direct.h"
#include "exp_put.h"
#include "exp_put_direct.h"
#include "exp_throughput.h"
#include "exp_update.h"
#include "ycsb_perf.h"

using namespace Component;
using namespace std;

boost::program_options::variables_map g_vm; 

boost::program_options::options_description g_desc("Options");

int main(int argc, char * argv[])
{
#ifdef PROFILE
  ProfilerDisable();
#endif

  namespace po = boost::program_options; 

  try {
    show_program_options();
    string filename;

    po::store(po::parse_command_line(argc, argv, g_desc), g_vm);

    if(g_vm.count("help")) {
      std::cout << g_desc;
      return 0;
    }

    props.setProperty("threads", g_vm["threads"].as<string>());
    props.setProperty("db", g_vm["db"].as<string>());
    props.setProperty("host", g_vm["host"].as<string>());
    props.setProperty("port", g_vm["port"].as<string>());
    props.setProperty("operation", g_vm["operation"].as<string>());
    props.setProperty("debug_level", g_vm.count("debug_level") > 0
                                         ? g_vm["debug_level"].as<string>()
                                         : "0");
    filename = g_vm["workload"].as<std::string>();
  }
  catch (const po::error &ex) {
    std::cerr << ex.what() << '\n';
    return -1;
  }

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
  ycsb::Workload wl;

  cpu_mask_t cpus;

  try
  {
    cpus = Experiment::get_cpu_mask_from_string(
        atoi(props.getProperty("threads")));
  }
  catch(...)
  {
    PERR("couldn't create CPU mask. Exiting.");
    return 1;
  }

#ifdef PROFILE
  ProfilerStart("cpu.profile");
#endif


#ifdef PROFILE
  ProfilerStop();
#endif
  
  return 0;
}


void show_program_options()
{
  namespace po = boost::program_options;

  g_desc.add_options()("help", "Show help")(
      "operation", po::value<std::string>(), "Operation <load/run>.")(
      "workload", po::value<std::string>(), "Workload file path.")(
      "threads", po::value<std::string>()->default_value("1"),
      "Number of client threads.")("db", po::value<std::string>(),
                                   "Database to test.")(
      "host", po::value<std::string>(), "Server IP address.")(
      "port", po::value<string>(), "Server port.")(
      "debug_level", po::value<string>()->default_value("0"),
      "Debug level. Default: 0.");
}

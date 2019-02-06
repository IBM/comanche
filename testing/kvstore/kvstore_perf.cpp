/* note: we do not include component source, only the API definition */
#include <common/utils.h>
#include <common/str_utils.h>
#include <core/task.h>
#include <api/components.h>
#include <api/kvstore_itf.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <iostream>

#define DEFAULT_PATH "/mnt/pmem0/"
#define POOL_NAME "test.pool"
#undef PROFILE

#ifdef PROFILE
#include <gperftools/profiler.h>
#endif


#include "data.h"
#include "exp_put.h"
#include "exp_get.h"
#include "exp_get_direct.h"
#include "exp_put_direct.h"
#include "exp_throughput.h"
#include "exp_update.h"
#include "kvstore_perf.h"

using namespace Component;

ProgramOptions Options;
Data * g_data;
int g_argc;
char ** g_argv;
uint64_t g_iops;

pthread_mutex_t g_write_lock = PTHREAD_MUTEX_INITIALIZER;
boost::program_options::options_description g_desc("Options");

int main(int argc, char * argv[])
{
#ifdef PROFILE
  ProfilerDisable();
#endif
  
  g_argc = argc;
  g_argv = argv;

  namespace po = boost::program_options; 

  try {
    show_program_options();

    po::variables_map vm; 
    po::store(po::parse_command_line(argc, argv, g_desc),  vm);

    if(vm.count("help")) {
      std::cout << g_desc;
      return 0;
    }

    Options.component = vm.count("component") > 0 ? vm["component"].as<std::string>() : DEFAULT_COMPONENT;
    Options.test = vm.count("test") > 0 ? vm["test"].as<std::string>() : "all";
    Options.cores  = vm.count("cores") > 0 ? vm["cores"].as<std::string>() : "0";
    Options.elements = vm.count("elements") > 0 ? vm["elements"].as<int>() : 100000;
    Options.key_length = vm.count("key_length") > 0 ? vm["key_length"].as<unsigned int>() : 8;
    Options.value_length = vm.count("value_length") > 0 ? vm["value_length"].as<unsigned int>() : 32;
    Options.skip_json_reporting = vm.count("skip_json_reporting");
    Options.pin = !(vm.count("nopin") > 0);
  }
  catch (const po::error &ex) {
    std::cerr << ex.what() << '\n';
    return -1;
  }

  bool use_direct_memory = Options.component == "dawn";
  g_data = new Data(Options.elements, Options.key_length, Options.value_length, use_direct_memory);

  Options.report_file_name = Experiment::create_report(Options);

  cpu_mask_t cpus;

  try
  {
    cpus = Experiment::get_cpu_mask_from_string(Options.cores);
  }
  catch(...)
  {
    PERR("couldn't create CPU mask. Exiting.");
    return 1;
  }

#ifdef PROFILE
  ProfilerStart("cpu.profile");
#endif

  if (Options.test == "all" || Options.test == "put") {
    Core::Per_core_tasking<ExperimentPut, ProgramOptions> exp(cpus, Options, Options.pin);
    exp.wait_for_all();

    auto first_exp = exp.tasklet(cpus.first_core());
    first_exp->summarize();
  }
  
  if (Options.test == "all" || Options.test == "get") {
    Core::Per_core_tasking<ExperimentGet, ProgramOptions> exp(cpus, Options, Options.pin);

    exp.wait_for_all();

    auto first_exp = exp.tasklet(cpus.first_core());
    first_exp->summarize();
  }
  
  if (Options.test == "all" || Options.test == "get_direct") {
    Core::Per_core_tasking<ExperimentGetDirect, ProgramOptions> exp(cpus, Options, Options.pin);
    exp.wait_for_all();

    auto first_exp = exp.tasklet(cpus.first_core());
    first_exp->summarize();
  }
  
  if (Options.test == "all" || Options.test == "put_direct") {
    Core::Per_core_tasking<ExperimentPutDirect, ProgramOptions> exp(cpus, Options, Options.pin);
    exp.wait_for_all();

    auto first_exp = exp.tasklet(cpus.first_core());
    first_exp->summarize();
  }
  
  if (Options.test == "all" || Options.test == "throughput") {
    Core::Per_core_tasking<ExperimentThroughput, ProgramOptions> exp(cpus, Options, Options.pin);
    exp.wait_for_all();
    auto first_exp = exp.tasklet(cpus.first_core());
    first_exp->summarize(); /* print aggregate IOPS */
  }

  if (Options.test == "all" || Options.test == "update") {
    Core::Per_core_tasking<ExperimentUpdate, ProgramOptions> exp(cpus, Options, Options.pin);
    exp.wait_for_all();

    auto first_exp = exp.tasklet(cpus.first_core());
    first_exp->summarize();
  }

#ifdef PROFILE
  ProfilerStop();
#endif
  
  return 0;
}


void show_program_options()
{
  namespace po = boost::program_options;

  g_desc.add_options()
    ("help", "Show help")
    ("test", po::value<std::string>(), "Test name <all|put|get|put_direct|get_direct>. Default: all.")
    ("component", po::value<std::string>()->default_value(DEFAULT_COMPONENT), "Implementation selection <filestore|pmstore|dawn|nvmestore|mapstore|hstore>. Default: filestore.")
    ("cores", po::value<std::string>(), "Cores to run tasks on. Supports singles and ranges. Example: a,b,c-d. Default: Core 0.")
    ("path", po::value<std::string>(), "Path of directory for pool. Default: current directory.")
    ("pool_name", po::value<std::string>(), "Prefix name of pool; will append core number. Default: Exp.pool")
    ("size", po::value<unsigned long long int>(), "Size of pool. Default: 100MB.")
    ("flags", po::value<int>(), "Flags for pool creation. Default: none.")
    ("elements", po::value<int>(), "Number of data elements. Default: 100,000.")
    ("key_length", po::value<unsigned int>(), "Key length of data. Default: 8.")
    ("value_length", po::value<unsigned int>(), "Value length of data. Default: 32.")
    ("bins", po::value<unsigned int>(), "Number of bins for statistics. Default: 100. ")
    ("latency_range_min", po::value<double>(), "Lowest latency bin threshold. Default: 10e-9.")
    ("latency_range_max", po::value<double>(), "Highest latency bin threshold. Default: 10e-3.")
    ("debug_level", po::value<int>(), "Debug level. Default: 0.")
    ("owner", po::value<std::string>(), "Owner name for component registration")
    ("server_address", po::value<std::string>(), "Server address, with port")
    ("device_name", po::value<std::string>(), "Device name")
    ("pci_addr", po::value<std::string>(), "PCI address (e.g. 0b:00.0)")
    ("nopin", "Do not pin down worker threads to cores")
    ("verbose", "Verbose output")
    ("summary", "Prints summary statement: most frequent latency bin info per core")
    ("skip_json_reporting", "disables creation of json report file")
    ;
}

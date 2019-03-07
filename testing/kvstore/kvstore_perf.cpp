/* note: we do not include component source, only the API definition */

#include "data.h"
#include "exp_put.h"
#include "exp_get.h"
#include "exp_get_direct.h"
#include "exp_put_direct.h"
#include "exp_throughput.h"
#include "exp_throughupdate.h"
#include "exp_update.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#include <common/utils.h>
#pragma GCC diagnostic pop
#include <common/str_utils.h>
#include "task.h"
#include <api/components.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <api/kvstore_itf.h>
#pragma GCC diagnostic pop
#include <boost/program_options.hpp>

#undef PROFILE

#ifdef PROFILE
#include <gperftools/profiler.h>
#endif

#include <chrono>
#include <iostream>
#include <mutex>
#include <numeric> /* accumulate */

#define DEFAULT_COMPONENT "filestore"

using namespace Component;

Data * g_data;
double g_iops;
boost::program_options::variables_map g_vm; 

std::mutex g_write_lock;

namespace
{
  boost::program_options::options_description g_desc("Options");
  boost::program_options::positional_options_description g_pos;

  template <typename Exp>
    void run_exp(cpu_mask_t cpus, const ProgramOptions &options)
  {
    Core::Per_core_tasking<Exp, ProgramOptions> exp(cpus, options, options.pin);
    exp.wait_for_all();
    auto first_exp = exp.tasklet(cpus.first_core());
    first_exp->summarize();
  }

  using exp_f = void(*)(cpu_mask_t, const ProgramOptions &);
  using test_element = std::pair<std::string, exp_f>;
  const std::vector<test_element> test_vector
  {
    { "put", run_exp<ExperimentPut> },
    { "get", run_exp<ExperimentGet> },
    { "get_direct", run_exp<ExperimentGetDirect> },
    { "put_direct", run_exp<ExperimentPutDirect> },
    { "throughput", run_exp<ExperimentThroughput> },
    { "throughupdate", run_exp<ExperimentThroughupdate> },
    { "update", run_exp<ExperimentUpdate> },
  };
}

void show_program_options();

int main(int argc, char * argv[])
{
#ifdef PROFILE
  ProfilerDisable();
#endif

  namespace po = boost::program_options; 

  try {
    show_program_options();

    po::store(po::command_line_parser(argc, argv).options(g_desc).positional(g_pos).run(), g_vm);

    if(g_vm.count("help")) {
      std::cout << g_desc;
      return 0;
    }

    ProgramOptions Options(g_vm);

    bool use_direct_memory = Options.component == "dawn";
    g_data = new Data(Options.elements, Options.key_length, Options.value_length, use_direct_memory);

    Options.report_file_name = Options.do_json_reporting ? Experiment::create_report(Options.component) : "";

    cpu_mask_t cpus;

    try
    {
      cpus = Experiment::get_cpu_mask_from_string(Options.cores);
    }
    catch(...)
    {
      PERR("%s", "couldn't create CPU mask. Exiting.");
      return 1;
    }

#ifdef PROFILE
    ProfilerStart("cpu.profile");
#endif

    for ( const auto &e : test_vector )
    {
      if ( Options.test == "all" || Options.test == e.first )
      {
        e.second(cpus, Options);
      }
    }
#ifdef PROFILE
    ProfilerStop();
#endif
  }
  catch (const po::error &ex) {
    std::cerr << ex.what() << '\n';
    return -1;
  }
  
  return 0;
}



void show_program_options()
{
  namespace po = boost::program_options;

  const std::string test_names =
    "Test name <"
      + std::accumulate(
          test_vector.begin()
          , test_vector.end()
          , test_element("all", nullptr)
          , [] (const test_element &a, const test_element &b) { return test_element(a.first + "|" + b.first, nullptr); }
        ).first
      + ">. Default: all."
    ;

  g_desc.add_options()
    ("help", "Show help")
    ("test" , po::value<std::string>()->default_value("all"), test_names.c_str())
    ("component", po::value<std::string>()->default_value(DEFAULT_COMPONENT), "Implementation selection <filestore|pmstore|dawn|nvmestore|mapstore|hstore>. Default: filestore.")
    ("cores", po::value<std::string>()->default_value("0"), "Comma-separated ranges of core indexes to use for test. A range may be specified by a single index, a pair of indexes separated by a hyphen, or an index followed by a colon followed by a count of additional indexes. These examples all specify cores 2 through 4 inclusive: '2,3,4', '2-4', '2:3'. Default: 0.")
    ("devices", po::value<std::string>(), "Comma-separated ranges of devices to use during test. Each identifier is a dotted pair of numa zone and index, e.g. '1.2'. For comaptibility with cores, a simple index number is accepted and implies numa node 0. These examples all specify device indexes 2 through 4 inclusive in numa node 0: '2,3,4', '0.2:3'. These examples all specify devices 2 thourgh 4 inclusive on numa node 1: '1.2,1.3,1.4', '1.2-1.4', '1.2:3'.  When using hstore, the actual dax device names are concatenations of the device_name option with <node>.<index> values specified by this option. In the node 0 example above, with device_name /dev/dax, the device paths are /dev/dax0.2 through /dev/dax0.4 inclusive. Default: the value of cores.")
    ("path", po::value<std::string>(), "Path of directory for pool. Default: current directory.")
    ("pool_name", po::value<std::string>(), "Prefix name of pool; will append core number. Default: Exp.pool")
    ("size", po::value<unsigned long long int>(), "Size of pool. Default: 100MB.")
    ("flags", po::value<int>(), "Flags for pool creation. Default: none.")
    ("elements", po::value<int>()->default_value(100000), "Number of data elements. Default: 100,000.")
    ("key_length", po::value<unsigned int>()->default_value(8), "Key length of data. Default: 8.")
    ("value_length", po::value<unsigned int>()->default_value(32), "Value length of data. Default: 32.")
    ("bins", po::value<unsigned int>(), "Number of bins for statistics. Default: 100. ")
    ("latency_range_min", po::value<double>(), "Lowest latency bin threshold. Default: 10e-9.")
    ("latency_range_max", po::value<double>(), "Highest latency bin threshold. Default: 10e-3.")
    ("debug_level", po::value<int>(), "Debug level. Default: 0.")
    ("owner", po::value<std::string>(), "Owner name for component registration")
    ("server", po::value<std::string>(), "Dawn server IP address")
    ("port", po::value<unsigned>(), "Dawn server port. Default 11911")
    ("port_increment", po::value<unsigned>(), "Port increment every N instances. Default none.")
    ("device_name", po::value<std::string>(), "Device name")
    ("pci_addr", po::value<std::string>(), "PCI address (e.g. 0b:00.0)")
    ("nopin", "Do not pin down worker threads to cores")
    ("start_time", po::value<std::string>(), "Delay start time of experiment until specified time (HH:MM, 24 hour format expected. Default: start immediately.")
    ("verbose", "Verbose output")
    ("summary", "Prints summary statement: most frequent latency bin info per core")
    ("skip_json_reporting", "disables creation of json report file")
    ("continuous", "enables never-ending execution, if possible")
    ;
}

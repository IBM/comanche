/* note: we do not include component source, only the API definition */

#include "data.h"
#include "exp_put.h"
#include "exp_get.h"
#include "exp_get_direct.h"
#include "exp_put_direct.h"
#include "exp_throughput.h"
#include "exp_update.h"
#include "get_cpu_mask_from_string.h"
#include "get_vector_from_string.h"
#include "program_options.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#include <common/utils.h>
#pragma GCC diagnostic pop
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
#include <thread>

using namespace Component;

namespace
{
  boost::program_options::options_description po_desc("Options");
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
  static const std::vector<test_element> test_vector
  {
    { "put", run_exp<ExperimentPut> },
    { "get", run_exp<ExperimentGet> },
    { "get_direct", run_exp<ExperimentGetDirect> },
    { "put_direct", run_exp<ExperimentPutDirect> },
    { "throughput", run_exp<ExperimentThroughput> },
    { "update", run_exp<ExperimentUpdate> },
  };
}

#if 0
namespace
{
  void _cpu_mask_add_core_wrapper(cpu_mask_t &mask, unsigned core_first, unsigned core_last, unsigned mac_cores)
  {
    if ( core_last < core_first )
    {
      std::ostringstream e;
      e << "invalid core range specified: start (" << core_first << ") > end (" << core_last << ")";
      PERR("%s.", e.str().c_str());
      throw std::runtime_error(e.str());
    }
    else if ( mac_cores < core_last )  // mac_cores is zero indexed
    {
      std::ostringstream e;
      e << "specified core end (-" << core_last << "exceeds physical core count. Valid range is [0.." << mac_cores << ")";
      PERR("%s", e.str().c_str());
      throw std::runtime_error(e.str());
    }

    try
    {
      for (unsigned core = core_first; core != core_last; ++core)
      {
        mask.add_core(core);
      }
    }
    catch ( const Exception &e )
    {
      PERR("failed while adding core to mask: %s.", e.cause());
      throw;
    }
    catch(...)
    {
      PERR("%s", "failed while adding core to mask.");
      throw;
    }
  }

  cpu_mask_t get_cpu_mask_from_string(std::string core_string)
  {
    auto cores = get_vector_from_string<int>(core_string);
    cpu_mask_t mask;
    int hardware_total_cores = std::thread::hardware_concurrency();

    for ( auto c : cores )
    {
      _cpu_mask_add_core_wrapper(mask, c, c+1, hardware_total_cores);
    }

    return mask;
  }
}
#endif

int main(int argc, char * argv[])
{
#ifdef PROFILE
  ProfilerDisable();
#endif

  namespace po = boost::program_options; 

  try
  {
    std::vector<std::string> test_names;
    std::transform(
      test_vector.begin()
      , test_vector.end()
      , std::back_inserter(test_names)
      , [] ( const test_element &e ) { return e.first; }
    );
    ProgramOptions::add_program_options(po_desc, test_names);

    boost::program_options::variables_map vm; 
    po::store(po::command_line_parser(argc, argv).options(po_desc).positional(g_pos).run(), vm);

    if ( vm.count("help") )
    {
      std::cout << po_desc;
      return 0;
    }

    ProgramOptions Options(vm);

    bool use_direct_memory = Options.component == "dawn";
    Experiment::g_data = new Data(Options.elements, Options.key_length, Options.value_length, use_direct_memory);

    Options.report_file_name = Options.do_json_reporting ? Experiment::create_report(Options.component) : "";

    cpu_mask_t cpus;

    try
    {
      cpus = get_cpu_mask_from_string(Options.cores);
    }
    catch (...)
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

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

#include <algorithm>
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
  const std::vector<test_element> test_vector
  {
    { "put", run_exp<ExperimentPut> },
    { "get", run_exp<ExperimentGet> },
    { "get_direct", run_exp<ExperimentGetDirect> },
    { "put_direct", run_exp<ExperimentPutDirect> },
    { "throughput", run_exp<ExperimentThroughput> },
    { "update", run_exp<ExperimentUpdate> },
  };
}

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

    if ( Options.test == "all" )
    {
      for ( const auto &e : test_vector )
      {
        e.second(cpus, Options);
      }
    }
    else
    {
      const auto it =
        std::find_if(
          test_vector.begin()
          , test_vector.end()
          , [&Options] (const test_element &a) { return a.first == Options.test; }
        );
      if ( it == test_vector.end() )
      {
        PERR("No such test: %s.", Options.test.c_str());
        return 1;
      }
      it->second(cpus, Options);
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

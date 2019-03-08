#include "program_options.h"

#include <algorithm>

namespace
{
  unsigned clamp(unsigned value_, unsigned min_, unsigned mac_)
  {
    return std::min(std::max(value_, min_), mac_);
  }
}

ProgramOptions::ProgramOptions(const boost::program_options::variables_map &vm_)
  : test(vm_["test"].as<std::string>())
  , component(vm_["component"].as<std::string>())
  , cores(vm_["cores"].as<std::string>())
  , elements(vm_["elements"].as<int>())
  , key_length(vm_["key_length"].as<unsigned>())
  , value_length(vm_["value_length"].as<unsigned>())
  , do_json_reporting( ! vm_.count("skip_json_reporting") )
  , pin( ! vm_.count("nopin") )
  , continuous( vm_.count("continuous") )
  , verbose( vm_.count("verbose") )
  , summary( vm_.count("summary") )
  , read_pct( clamp(vm_["read_pct"].as<unsigned>(), 0U, 100U) )
  , devices()
  , time_secs()
  , path()
  , pool_name()
  , size()
  , flags()
  , report_file_name()
  , bin_count()
  , bin_threshold_min()
  , bin_threshold_max()
  , debug_level()
  , start_time()
  , owner()
  , server_address()
  , device_name()
  , pci_addr()
{}

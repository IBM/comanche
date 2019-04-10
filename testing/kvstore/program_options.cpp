#include "program_options.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <api/kvstore_itf.h>
#pragma GCC diagnostic pop
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#include <common/logging.h>
#pragma GCC diagnostic pop
#include <common/str_utils.h> /* MB */
#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip> /* get_time */
#include <numeric> /* accumulate */
#include <stdexcept>
#include <string>

#define DEFAULT_COMPONENT "filestore"

namespace
{
  unsigned clamp(unsigned value_, unsigned min_, unsigned mac_)
  {
    return std::min(std::max(value_, min_), mac_);
  }
}

std::chrono::system_clock::time_point parse_local_hh_mm(const std::string &t_)
{
  std::istringstream time_string(t_);
  std::tm time_input;
  time_string >> std::get_time(&time_input, "%R");
  if ( time_string.fail() )
  {
    auto e = "Error reading time string " + t_ + ". Should be HH:MM format (24 hour clock)";
    PERR("%s.", e.c_str());
    throw std::domain_error(e);
  }

  const time_t now = time(NULL);
  std::tm * now_tm = localtime(&now);
  std::tm go_tm;
  std::memcpy(&go_tm, now_tm, sizeof(std::tm));
  go_tm.tm_min = time_input.tm_min;
  go_tm.tm_hour = time_input.tm_hour;

  PINF("delaying experiment start until %d:%.2d", go_tm.tm_hour, go_tm.tm_min);

  return std::chrono::system_clock::from_time_t(mktime(&go_tm));
}


ProgramOptions::ProgramOptions(const boost::program_options::variables_map &vm_)
try
  : test(vm_["test"].as<std::string>())
  , component(vm_["component"].as<std::string>())
  , cores(vm_["cores"].as<std::string>())
  , elements(vm_["elements"].as<std::size_t>())
  , key_length(vm_["key_length"].as<unsigned>())
  , value_length(vm_["value_length"].as<unsigned>())
  , do_json_reporting( ! vm_.count("skip_json_reporting") )
  , pin( ! vm_.count("nopin") )
  , continuous( vm_.count("continuous") )
  , verbose( vm_.count("verbose") )
  , summary( vm_.count("summary") )
  , read_pct( clamp(vm_["read_pct"].as<unsigned>(), 0U, 100U) )
  , insert_erase_pct( clamp(vm_["insert_erase_pct"].as<unsigned>(), 0U, 100U) )
  , devices(vm_.count("devices") ? vm_["devices"].as<std::string>() : cores)
  , time_secs()
  , path( vm_.count("path") ? vm_["path"].as<std::string>() : boost::optional<std::string>() )
  , pool_name( vm_.count("pool_name") > 0 ? vm_["pool_name"].as<std::string>() : "" )
  , size(vm_["size"].as<unsigned long long int>())
  , flags(vm_["flags"].as<std::uint32_t>())
  , report_file_name()
  , bin_count(vm_["bins"].as<unsigned>())
  , bin_threshold_min(vm_["latency_range_min"].as<double>())
  , bin_threshold_max(vm_["latency_range_max"].as<double>())
  , debug_level(vm_["debug_level"].as<int>())
  , start_time(vm_.count("start_time") ? parse_local_hh_mm(vm_["start_time"].as<std::string>()) : boost::optional<std::chrono::system_clock::time_point>() )
  , duration(vm_.count("duration") ? vm_["duration"].as<unsigned>() : boost::optional<unsigned>() )
  , owner(vm_["owner"].as<std::string>())
  , server_address(vm_["server"].as<std::string>())
  , port(vm_["port"].as<unsigned>())
  , port_increment( vm_.count("port_increment") ? vm_["port_increment"].as<unsigned>() : boost::optional<unsigned>())
  , device_name(vm_.count("device_name") ? vm_["device_name"].as<std::string>() : boost::optional<std::string>())
  , pci_addr( vm_.count("pci_addr_valid") ? vm_["pci_addr"].as<std::string>() : boost::optional<std::string>() )
{
  if ( ( component_is("pmstore") || component_is("hstore") ) && ! path )
  {
    auto e = "component '" + component + "' requires --path argument for persistent memory store";
    throw std::runtime_error(e);
  }

  if ( component_is( "nvmestore" ) && ! pci_addr )
  {
    auto e = "component '" + component + "' requires --pci_addr argument";
    throw std::runtime_error(e);
  }
}
catch ( const std::exception &ex )
{
  PERR("%s. Aborting!", ex.what());
  throw;
}

void ProgramOptions::add_program_options(
  boost::program_options::options_description &desc_
  , const std::vector<std::string> &test_names_
)
{
  namespace po = boost::program_options;

  const std::string test_names =
    "Test name <"
      + std::accumulate(
          test_names_.begin()
          , test_names_.end()
          , std::string("all")
          , [] (const std::string &a, const std::string &b) { return a + "|" + b; }
        )
      + ">. Default: all."
    ;

  desc_.add_options()
    ("help", "Show help")
    ("test" , po::value<std::string>()->default_value("all"), test_names.c_str())
    ("component", po::value<std::string>()->default_value(DEFAULT_COMPONENT), "Implementation selection <filestore|pmstore|dawn|nvmestore|mapstore|hstore>. Default: filestore.")
    ("cores", po::value<std::string>()->default_value("0"), "Comma-separated ranges of core indexes to use for test. A range may be specified by a single index, a pair of indexes separated by a hyphen, or an index followed by a colon followed by a count of additional indexes. These examples all specify cores 2 through 4 inclusive: '2,3,4', '2-4', '2:3'. Default: 0.")
    ("devices", po::value<std::string>(), "Comma-separated ranges of devices to use during test. Each identifier is a dotted pair of numa zone and index, e.g. '1.2'. For comaptibility with cores, a simple index number is accepted and implies numa node 0. These examples all specify device indexes 2 through 4 inclusive in numa node 0: '2,3,4', '0.2:3'. These examples all specify devices 2 thourgh 4 inclusive on numa node 1: '1.2,1.3,1.4', '1.2-1.4', '1.2:3'.  When using hstore, the actual dax device names are concatenations of the device_name option with <node>.<index> values specified by this option. In the node 0 example above, with device_name /dev/dax, the device paths are /dev/dax0.2 through /dev/dax0.4 inclusive. Default: the value of cores.")
    ("path", po::value<std::string>()->default_value("./data/"), "Path of directory for pool. Default: \"./data/\"")
    ("pool_name", po::value<std::string>()->default_value("Exp.pool"), "Prefix name of pool; will append core number. Default: \"Exp.pool\"")
    ("size", po::value<unsigned long long>()->default_value(MB(100)), "Size of pool. Default: 100MiB.")
    ("flags", po::value<std::uint32_t>()->default_value(Component::IKVStore::FLAGS_SET_SIZE), "Flags for pool creation. Default: Component::IKVStore::FLAGS_SET_SIZE.")
    ("elements", po::value<std::size_t>()->default_value(100000), "Number of data elements. Default: 100,000.")
    ("key_length", po::value<unsigned>()->default_value(8), "Key length of data. Default: 8.")
    ("value_length", po::value<unsigned>()->default_value(32), "Value length of data. Default: 32.")
    ("bins", po::value<unsigned>()->default_value(100), "Number of bins for statistics. Default: 100. ")
    ("latency_range_min", po::value<double>()->default_value(0.000000001), "Lowest latency bin threshold. Default: 1e-9.")
    ("latency_range_max", po::value<double>()->default_value(0.001), "Highest latency bin threshold. Default: 1e-3.")
    ("debug_level", po::value<int>()->default_value(0), "Debug level. Default: 0.")
    ("read_pct", po::value<unsigned>()->default_value(0) , "Read percentage in throughput test. Default: 0.")
    ("insert_erase_pct", po::value<unsigned>()->default_value(0) , "Insert/erase percentage in throughput test. Default: 0.")
    ("owner", po::value<std::string>()->default_value("owner"), "Owner name for component registration")
    ("server", po::value<std::string>()->default_value("127.0.0.1"), "Dawn server IP address. Default: 127.0.0.1")
    ("port", po::value<unsigned>()->default_value(11911), "Dawn server port. Default 11911")
    ("port_increment", po::value<unsigned>(), "Port increment every N instances.")
    ("device_name", po::value<std::string>()->default_value("unused"), "Device name.")
    ("pci_addr", po::value<std::string>()->default_value("no_pci_addr"), "PCI address (e.g. 0b:00.0).")
    ("nopin", "Do not pin down worker threads to cores")
    ("start_time", po::value<std::string>(), "Delay start time of experiment until specified time (HH:MM, 24 hour format expected.")
    ("verbose", "Verbose output")
    ("summary", "Prints summary statement: most frequent latency bin info per core")
    ("skip_json_reporting", "disables creation of json report file")
    ("continuous", "enables never-ending execution, if possible")
    ("duration", po::value<unsigned>(), "throughput test duration, in seconds")
    ;
}

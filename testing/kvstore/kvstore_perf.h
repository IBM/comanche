#ifndef __KVSTORE_PERF_H__
#define __KVSTORE_PERF_H__

#include <api/kvstore_itf.h>
#include <pthread.h>
#include <sys/stat.h>

#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"

#define DEFAULT_COMPONENT "filestore"
#define PMSTORE_PATH "libcomanche-pmstore.so"
#define FILESTORE_PATH "libcomanche-storefile.so"
#define NVMESTORE_PATH "libcomanche-nvmestore.so"
#define ROCKSTORE_PATH "libcomanche-rocksdb.so"
#define DAWN_PATH "libcomanche-dawn-client.so"

boost::program_options::options_description desc("Options");
void show_program_options();

struct ProgramOptions {
  std::string test;
  std::string component = DEFAULT_COMPONENT;
  std::string cores;
  unsigned time_secs;
  std::string path;
  std::string pool_name;
  unsigned long long int size;
  int flags;
  int elements;
  std::string report_file_name;
  unsigned int key_length;
  unsigned int value_length;
  unsigned int bin_count;
  double bin_threshold_min;
  double bin_threshold_max;
  int debug_level;
  bool summary;
  std::string owner;
  std::string server_address;
  std::string device_name;
  std::string pci_addr;
}; 

void show_program_options()
{
  namespace po = boost::program_options;

  desc.add_options()
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
    ("value_length", po::value<unsigned int>(), "Value length of data. Default: 64.")
    ("bins", po::value<unsigned int>(), "Number of bins for statistics. Default: 100. ")
    ("latency_range_min", po::value<double>(), "Lowest latency bin threshold. Default: 10e-9.")
    ("latency_range_max", po::value<double>(), "Highest latency bin threshold. Default: 10e-3.")
    ("debug_level", po::value<int>(), "Debug level. Default: 0.")
    ("owner", po::value<std::string>(), "Owner name for component registration")
    ("server_address", po::value<std::string>(), "Server address, with port")
    ("device_name", po::value<std::string>(), "Device name")
    ("pci_addr", po::value<std::string>(), "PCI address (e.g. 0b:00.0)")
    ("verbose", "Verbose output")
    ("summary", "Prints summary statement: most frequent latency bin info per core")
    ;
}

#endif

#ifndef __KVSTORE_PERF_H__
#define __KVSTORE_PERF_H__

#include <string>

#define DEFAULT_COMPONENT "filestore"
#define PMSTORE_PATH "libcomanche-pmstore.so"
#define FILESTORE_PATH "libcomanche-storefile.so"
#define NVMESTORE_PATH "libcomanche-nvmestore.so"
#define ROCKSTORE_PATH "libcomanche-rocksdb.so"
#define DAWN_PATH "libcomanche-dawn-client.so"

void show_program_options();

struct ProgramOptions {
  std::string test;
  std::string component = DEFAULT_COMPONENT;
  std::string cores;
  std::string devices;
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
  bool skip_json_reporting;
  bool pin;
  std::string start_time;
  std::string owner;
  std::string server_address;
  std::string device_name;
  std::string pci_addr;
  ProgramOptions()
    : test()
    , cores()
    , devices()
    , time_secs()
    , path()
    , pool_name()
    , size()
    , flags()
    , elements()
    , report_file_name()
    , key_length()
    , value_length()
    , bin_count()
    , bin_threshold_min()
    , bin_threshold_max()
    , debug_level()
    , summary()
    , skip_json_reporting()
    , pin()
    , start_time()
    , owner()
    , server_address()
    , device_name()
    , pci_addr()
  {}
}; 


#endif

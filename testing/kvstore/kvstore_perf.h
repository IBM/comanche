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

extern boost::program_options::options_description g_desc;

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


#endif

#ifndef __KVSTORE_PROGRAM_OPTIONS_H__
#define __KVSTORE_PROGRAM_OPTIONS_H__

#include <boost/program_options.hpp>

#include <string>

struct ProgramOptions
{
  /* finalized in constructor */
  std::string test;
  std::string component;
  std::string cores;
  int elements;
  unsigned int key_length;
  unsigned int value_length;
  bool do_json_reporting;
  bool pin;
  bool continuous;
  bool verbose;
  bool summary;
  /* finalized later */
  std::string devices;
  unsigned time_secs;
  std::string path;
  std::string pool_name;
  unsigned long long int size;
  int flags;
  std::string report_file_name;
  unsigned int bin_count;
  double bin_threshold_min;
  double bin_threshold_max;
  int debug_level;
  std::string start_time;
  std::string owner;
  std::string server_address;
  std::string device_name;
  std::string pci_addr;
  ProgramOptions(const boost::program_options::variables_map &);
}; 

#endif

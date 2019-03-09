#ifndef __KVSTORE_PROGRAM_OPTIONS_H__
#define __KVSTORE_PROGRAM_OPTIONS_H__

#include <boost/optional.hpp>
#include <boost/program_options.hpp>

#include <chrono>
#include <string>
#include <vector>

class ProgramOptions
{
private:
  bool component_is(const std::string &c) const { return component == c; }
public:
  /* finalized in constructor */
  std::string test;
  std::string component;
  std::string cores;
  std::size_t elements;
  unsigned key_length;
  unsigned value_length;
  bool do_json_reporting;
  bool pin;
  bool continuous;
  bool verbose;
  bool summary;
  unsigned read_pct;
  /* finalized later */
  std::string devices;
  unsigned time_secs;
  boost::optional<std::string> path;
  std::string pool_name;
  unsigned long long size;
  std::uint32_t flags;
  std::string report_file_name;
  unsigned bin_count;
  double bin_threshold_min;
  double bin_threshold_max;
  int debug_level;
  boost::optional<std::chrono::system_clock::time_point> start_time;
  std::string owner;
  std::string server_address;
  unsigned port;
  boost::optional<unsigned> port_increment;
  boost::optional<std::string> device_name;
  boost::optional<std::string> pci_addr;

  ProgramOptions(const boost::program_options::variables_map &);

  static void add_program_options(
    boost::program_options::options_description &desc
    , const std::vector<std::string> &test_vector
  );
}; 

#endif

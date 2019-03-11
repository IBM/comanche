#ifndef __EXPERIMENT_H__
#define __EXPERIMENT_H__

#include "task.h"

#include "dotted_pair.h"
#include "statistics.h"
#include "stopwatch.h"

#include "rapidjson/document.h"

#include <boost/optional.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <api/kvstore_itf.h>
#pragma GCC diagnostic pop

#include <chrono>
#include <map> /* map - should follow local includes */
#include <cstddef> /* size_t */
#include <string>
#include <mutex>
#include <vector>

class Data;
class ProgramOptions;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
class Experiment : public Core::Tasklet
{
private:
  std::string _pool_path;
  std::string _pool_name;
  std::string _pool_name_local;  // full pool name, e.g. "Exp.pool.0" for core 0
  std::string _owner;
  unsigned long long int _pool_size;
  std::uint32_t _pool_flags;
  std::size_t _pool_num_objects;
  std::string _cores;
  std::string _devices;
  std::vector<int> _core_list;
  int _execution_time;
  boost::optional<std::chrono::system_clock::time_point> _start_time; // default behavior: start now
  int _debug_level;
  std::string _component;
  std::string _results_path;
  std::string _report_filename;
  bool _do_json_reporting;
  std::string _test_name;
  Component::IKVStore::memory_handle_t _memory_handle;

  // common experiment parameters
  Component::IKVStore *                 _store;
  Component::IKVStore::pool_t           _pool;
  std::string                           _server_address;
  unsigned                              _port;
  unsigned                              _port_increment;
  boost::optional<std::string>          _device_name;
  boost::optional<std::string>          _pci_address;
public:
  bool                                  _first_iter = true;
private:
  bool                                  _ready = false;
public:
  Stopwatch timer;
private:
  bool _verbose;
  bool _summary;

  // member variables for tracking pool sizes
  long _element_size_on_disk = -1; // floor: filesystem block size
  long _element_size = -1; // raw amount of file data (bytes)
  long _elements_in_use = 0;
  long _pool_element_start = 0;
public:
  long _pool_element_end = -1;
private:
  long _elements_stored = 0;
  unsigned long _total_data_processed = 0;  // for use with throughput calculation

  // bin statistics
  int _bin_count;
  double _bin_threshold_min;
  double _bin_threshold_max;
  double _bin_increment;

  using core_to_device_map_t = std::map<unsigned, dotted_pair<unsigned>>;
  core_to_device_map_t _core_to_device_map;

  static core_to_device_map_t make_core_to_device_map(const std::string &cores, const std::string &devices);
  
public:
  static Data * g_data;
  static std::mutex g_write_lock;
  static double g_iops;

  Experiment(std::string name_, const ProgramOptions &options);

  Experiment(const Experiment &) = delete;
  Experiment& operator=(const Experiment &) = delete;

  virtual ~Experiment() {
    _store->release_ref();
  }

  auto test_name() const -> std::string { return _test_name; }
  int bin_count() const { return _bin_count; }
  double bin_threshold_min() const { return _bin_threshold_min; }
  double bin_threshold_max() const { return _bin_threshold_max; }
  Component::IKVStore *store() const { return _store; }
  Component::IKVStore::pool_t pool() const { return _pool; }
  std::size_t pool_num_objects() const { return _pool_num_objects; }
  bool is_verbose() const { return _verbose; }
  bool is_summary() const { return _summary; }
  unsigned long total_data_processed() const { return _total_data_processed; }
  bool is_json_reporting() const { return _do_json_reporting; }
  long pool_element_start() const { return _pool_element_start; }
  long pool_element_end() const { return _pool_element_end; }
  bool component_is(const std::string &c) const { return _component == c; }
  unsigned long long pool_size() const { return _pool_size; }
  Component::IKVStore::memory_handle_t memory_handle() const { return _memory_handle; }

  void initialize(unsigned core) override;

  // if experiment should be delayed, stop here and wait. Otherwise, start immediately
  void wait_for_delayed_start(unsigned core);

  /* maximun size of any dax device. Limitation: considers only dax devices specified in the device string */
  std::size_t dev_dax_max_size(const std::string & dev_dax_prefix_);

  /* maximun number of dax devices in any numa node.
   * Limitations:
   *   Assumes /dev/dax<node>.<number> representation)
   *   Considers only dax devices specified in the device string.
   */
  unsigned dev_dax_max_count_per_node(const std::string & /* dev_dax_prefix_ */);

  dotted_pair<unsigned> core_to_device(int core);

  int initialize_store(unsigned core);

  virtual void initialize_custom(unsigned core);

  bool ready() override
  {
    return _ready;
  }

  virtual void cleanup_custom(unsigned core);

  void _update_aggregate_iops(double iops);

  void summarize();

  void cleanup(unsigned core) noexcept override;

  bool component_uses_direct_memory()
  {
    return component_is( "dawn" );
  }

  bool component_uses_rdma()
  {
    return component_is( "dawn" );
  }

  void _debug_print(unsigned core, std::string text, bool limit_to_core0=false);

  unsigned _get_core_index(unsigned core) ;

  rapidjson::Document _get_report_document() ;

  void _initialize_experiment_report(rapidjson::Document& document) ;

  void _report_document_save(rapidjson::Document& document, unsigned core, rapidjson::Value& new_info) ;

  void _print_highest_count_bin(BinStatistics& stats, unsigned core);

  rapidjson::Value _add_statistics_to_report(std::string /* name */, BinStatistics& stats, rapidjson::Document& document) ;

  BinStatistics _compute_bin_statistics_from_vectors(std::vector<double> data, std::vector<double> data_bins, int bin_count, double bin_min, double bin_max, std::size_t elements) ;

  BinStatistics _compute_bin_statistics_from_vector(std::vector<double> data, int bin_count, double bin_min, double bin_max) ;

  /* create_report: output a report in JSON format with experiment data
   * Report format:
   *      experiment object - contains experiment parameters
   *      data object - actual results
   */
  static std::string create_report(const std::string component_);

  long GetDataInputSize(std::size_t index);

  unsigned long GetElementSize(unsigned core, std::size_t index) ;

  void _update_data_process_amount(unsigned core, std::size_t index) ;

  // throughput = Mib/s here
  double _calculate_current_throughput() ;

  void _populate_pool_to_capacity(unsigned core, Component::IKVStore::memory_handle_t memory_handle = Component::IKVStore::HANDLE_NONE) ;

  // assumptions: i_ is tracking current element in use
  void _enforce_maximum_pool_size(unsigned core, std::size_t i_) ;

  void _erase_pool_entries_in_range(std::size_t start, std::size_t finish) ;

};
#pragma GCC diagnostic pop


#endif //  __EXPERIMENT_H__

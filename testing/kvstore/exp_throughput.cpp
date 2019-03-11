#include "exp_throughput.h"

#include "data.h"
#include "program_options.h"

#include "boost/date_time/posix_time/posix_time.hpp"

#include <csignal>

std::mutex ExperimentThroughput::_iops_lock;
unsigned long ExperimentThroughput::_iops;
bool ExperimentThroughput::_stop= false;

std::uniform_int_distribution<int> distribution(0,99);

ExperimentThroughput::ExperimentThroughput(const ProgramOptions &options)
  : Experiment("throughput", options)
  , _i_rd(0)
  , _i_wr(0)
  , _start_time()
  , _report_time()
  , _report_interval(std::chrono::seconds(5))
  , _rand_engine()
  , _rand_pct(0, 99)
  , _rd_pct(options.read_pct)
  , _op_count_rd(0)
  , _op_count_wr(0)
  , _op_count_interval_rd(0)
  , _op_count_interval_wr(0)
  , _sw_rd()
  , _sw_wr()
  , _continuous(options.continuous)
{
}

void ExperimentThroughput::handler(int)
{
  _stop = true;
}

void ExperimentThroughput::initialize_custom(unsigned /* core */)
{
  _start_time = std::chrono::high_resolution_clock::now();
  if ( _continuous )
  {
    std::signal(SIGINT, handler);
  }
}

bool ExperimentThroughput::do_work(unsigned core)
{
  // handle first time setup
  if ( _first_iter )
  {
    wait_for_delayed_start(core);

    PLOG("[%u] Starting Throughput experiment (value len:%lu)...", core, g_data->value_len());
    _first_iter = false;

    /* DAX inifialization is serialized by thread (due to libpmempool behavior).
     * It is in some sense unfair to start measurement in initialize_custom,
     * which occurs before DAX initialization. Reset start of measurement to first
     * put operation.
     */
    _start_time = std::chrono::high_resolution_clock::now();
    _report_time = _start_time;
  }

  if ( pool_num_objects() <= _op_count_wr && _rand_pct(_rand_engine) < _rd_pct )
  {
    void * pval = 0;
    size_t pval_len;
    {
      StopwatchInterval si(_sw_rd);
      auto rc = store()->get(pool(), g_data->key(_i_rd), pval, pval_len);
      assert(rc == S_OK);
    }
    store()->free_memory(pval);
    ++_op_count_rd;
    ++_op_count_interval_rd;
  }
  else
  {
    {
      StopwatchInterval si(_sw_wr);
      auto rc = store()->put(pool(), g_data->key_as_string(_i_wr), g_data->value(_i_wr), g_data->value_len());
      assert(rc == S_OK);
    }
    ++_op_count_wr;
    ++_op_count_interval_wr;
    ++_i_wr;
  }

  auto now = std::chrono::high_resolution_clock::now();
  if ( _report_interval <= now - _report_time )
  {
    auto ptime = boost::posix_time::microsec_clock::universal_time();
    auto ptime_str = to_iso_extended_string(ptime);

    double secs = to_seconds(now - _report_time);
    unsigned long iops = static_cast<unsigned long>(double(_op_count_interval_rd + _op_count_interval_wr) / secs);
    PLOG(
      "time %s core %u IOps %lu"
      , ptime_str.c_str()
      , core
      , iops
    );
    _report_time += _report_interval;
    _op_count_interval_rd = 0;
    _op_count_interval_wr = 0;
    ++_i_rd;
  }

  if ( _continuous )
  {
    _i_rd %= pool_num_objects();
    _i_wr %= pool_num_objects();
  }

  auto do_more = _i_wr != pool_num_objects() && ! _stop;

  if ( ! do_more )
  {
    PINF("[%u] throughput: reached total number of objects. Exiting.", core);
  }

  return do_more;
}

std::chrono::high_resolution_clock::duration ExperimentThroughput::elapsed(std::chrono::high_resolution_clock::time_point p)
{
  return p - _start_time;
}

double ExperimentThroughput::to_seconds(std::chrono::high_resolution_clock::duration d)
{
  return double(std::chrono::duration_cast<std::chrono::milliseconds>(d).count()) / 1000.0;
}

void ExperimentThroughput::cleanup_custom(unsigned core)
{
  auto duration = elapsed(std::chrono::high_resolution_clock::now());
  _sw_rd.stop();
  _sw_wr.stop();

  PLOG("stopwatch : %g secs", _sw_rd.get_time_in_seconds() + _sw_wr.get_time_in_seconds());
  double secs = to_seconds(duration);
  PLOG("wall clock: %g secs", secs);
  PLOG("op count : rd %lu wr %lu", _op_count_rd, _op_count_wr);

  unsigned long iops = static_cast<unsigned long>(double(_op_count_rd + _op_count_wr) / secs);
  PLOG("core %u IOps %lu", core, iops);

  {
    std::lock_guard<std::mutex> g(_iops_lock);
    _iops += iops;
  }
}

void ExperimentThroughput::summarize()
{
  PMAJOR("total IOPS: %lu", _iops);
}

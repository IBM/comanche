#include "exp_throughupdate.h"

#include <csignal>

std::mutex ExperimentThroughupdate::_iops_lock;
unsigned long ExperimentThroughupdate::_iops;
bool ExperimentThroughupdate::_stop= false;

ExperimentThroughupdate::ExperimentThroughupdate(const ProgramOptions &options)
  : Experiment("throughupdate", options)
  , _start_time()
  , _end_time()
  , _report_time()
  , _report_interval(std::chrono::seconds(5))
  , _op_count(0)
  , _op_count_interval(0)
  , _sw()
  , _continuous(options.continuous)
{
}

void ExperimentThroughupdate::handler(int)
{
  _stop = true;
}

void ExperimentThroughupdate::initialize_custom(unsigned /* core */)
{
  _start_time = std::chrono::high_resolution_clock::now();
  if ( _continuous )
  {
    std::signal(SIGINT, handler);
  }
}

bool ExperimentThroughupdate::do_work(unsigned core)
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

  // assert(g_data);

  _sw.start();
  auto rc = store()->put(pool(), g_data->key_as_string(_i), g_data->value(_i), g_data->value_len());
  _sw.stop();

  assert(rc == S_OK);

  ++_i;
  ++_op_count;
  ++_op_count_interval;

  auto now = std::chrono::high_resolution_clock::now();
  if ( _report_interval <= now - _report_time )
  {
    double secs = to_seconds(now - _report_time);
    unsigned long iops = static_cast<unsigned long>(double(_op_count_interval) / secs);
    PLOG(
      "time %.3f IOps %lu core %u"
      , to_seconds(elapsed(now))
      , iops
      , core
    );
    _report_time += _report_interval;
    _op_count_interval = 0;
  }

  if ( _continuous )
  {
    _i %= pool_num_objects();
  } 
  if ( _i == pool_num_objects() )
  {
    PINF("[%u] throughput: reached total number of objects. Exiting.", core);
  }

  return _i != pool_num_objects() && ! _stop;
}

std::chrono::high_resolution_clock::duration ExperimentThroughupdate::elapsed(std::chrono::high_resolution_clock::time_point p)
{
  return p - _start_time;
}

double ExperimentThroughupdate::to_seconds(std::chrono::high_resolution_clock::duration d)
{
  return double(std::chrono::duration_cast<std::chrono::milliseconds>(d).count()) / 1000.0;
}

void ExperimentThroughupdate::cleanup_custom(unsigned core)
{
  auto duration = elapsed(std::chrono::high_resolution_clock::now());
  _sw.stop();

  PLOG("stopwatch : %g secs", _sw.get_time_in_seconds());
  double secs = to_seconds(duration);
  PLOG("wall clock: %g secs", secs);
  PLOG("op count : %lu", _op_count);

  unsigned long iops = static_cast<unsigned long>(double(_op_count) / secs);
  PLOG("%lu iops (core=%u)", iops, core);
  
  {
    std::lock_guard<std::mutex> g(_iops_lock);
    _iops += iops;
  }
}

void ExperimentThroughupdate::summarize()
{
  PMAJOR("total IOPS: %lu", _iops);
}

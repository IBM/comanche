/*
   Copyright [2017-2019] [IBM Corporation]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

/*
 * Authors:
 *
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#ifndef __CORE_TASK_H__
#define __CORE_TASK_H__

#include <common/cpu.h>
#include <common/errors.h>
#include <common/exceptions.h>
#include <common/spinlocks.h>
#include <common/utils.h>
#include <numa.h>
#include <sched.h>
#include <unistd.h>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace Core
{
/**
 * Basic unit of work
 *
 */
class Tasklet {
 public:
  virtual void initialize(unsigned core) = 0; /*< called once */
  virtual bool do_work(unsigned core) = 0; /*< called in tight loop; return false to exit */
  virtual void cleanup(unsigned core) = 0; /*< called once */
  virtual bool ready() { return true; }
};

/**
 * Manages per-core threads executing tasklets.
 *
 */
template <typename __Tasklet_t, typename __Arg_t>
class Per_core_tasking {
  static constexpr unsigned MAX_CORES = 256;
  static constexpr bool option_DEBUG = false;

 public:
  Per_core_tasking(cpu_mask_t& cpus, __Arg_t arg, bool pin=true) : _pin(pin) {
    for (unsigned c = 0; c < sysconf(_SC_NPROCESSORS_ONLN); c++) {
      if (cpus.check_core(c)) {
        _tasklet[c] = new __Tasklet_t(arg);
        _threads[c] = new std::thread(&Per_core_tasking::thread_entry, this, c);
      }
      else {
        _tasklet[c] = nullptr;
        _threads[c] = nullptr;
      }
    }

    for (unsigned c = 0; c < sysconf(_SC_NPROCESSORS_ONLN); c++) {
      if (cpus.check_core(c)) {
        while (!_tasklet[c]->ready()) usleep(100);
      }
    }

    _start_flag = true;
  }

  virtual ~Per_core_tasking() {
    _exit_flag = true;
    wait_for_all();
    if (option_DEBUG) PLOG("Per_core_tasking: threads joined");
  }

  void wait_for_all() {
    unsigned remaining;
    do {
      remaining = 0;
      
      for (unsigned c = 0; c < sysconf(_SC_NPROCESSORS_ONLN); c++) {
        if (_threads[c]) {
          void * rv;
          if(pthread_tryjoin_np(_threads[c]->native_handle(),&rv) == 0) {
            // delete _threads[c]; /* cannot delete after tryjoin? */
            _threads[c] = nullptr;
            delete _tasklet[c];
          }
          else remaining ++;
        }
      }      
    } while(remaining > 0);
  }

  __Tasklet_t* tasklet(unsigned core) {
    if (core >= MAX_CORES) throw General_exception("out of bounds");
    return _tasklet[core];
  }

 private:
  void thread_entry(unsigned core) {
    if(_pin)
      set_cpu_affinity(1UL << core);

    _tasklet[core]->initialize(core);

    while (!_start_flag) usleep(100);

    while (!_exit_flag) {
      try {
        if (!(_tasklet[core]->do_work(core)))
          break; /* if do_work return false, we exit the thread */
      } catch ( const Exception &e ) {
        PERR("do_work threw exception: %s", e.cause());
        break;
      } catch ( const std::exception &e ) {
        PERR("do_work threw exception %s", e.what());
        break;
      } catch (...) {
        PERR("do_work threw exception");
        break;
      }
    }

    _tasklet[core]->cleanup(core);
  }

 private:
  bool _start_flag = false;
  bool _exit_flag = false;
  const bool _pin;
  
  std::thread* _threads[MAX_CORES];
  __Tasklet_t* _tasklet[MAX_CORES];
};

}  // namespace Core

#endif

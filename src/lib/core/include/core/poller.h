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

#ifndef __CORE_POLLER_H__
#define __CORE_POLLER_H__

#include <common/cpu.h>
#include <common/errors.h>
#include <common/exceptions.h>
#include <common/spinlocks.h>
#include <common/utils.h>
#include <numa.h>
#include <sched.h>
#include <signal.h>
#include <unistd.h>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace Core
{
class Poller {
  static constexpr unsigned MAX_CORES = 128;
  static constexpr bool MASK_PROFILING_SIGNAL = true;

 public:
  Poller(cpu_mask_t& cpus) {
    for (unsigned c = 0; c < sysconf(_SC_NPROCESSORS_ONLN); c++) {
      if (cpus.check_core(c)) {
        _threads.push_back(new std::thread(&Poller::thread_entry, this, c));
        _cpu_mask.add_core(c);
        PLOG("Poller: created thread on core %u", c);
      }
    }
  }

  virtual ~Poller() {
    signal_exit();
    for (auto& t : _threads) {
      t->join();
      PLOG("Poller: thread exited %p", t);
      delete t;
    }
  }

  void signal_exit() { _exit_flag = true; }

  void register_percore_task(std::function<void*(unsigned, void*)> init_fp,
                             std::function<void(unsigned, void*)> callback_fp,
                             std::function<void(unsigned, void*)> cleanup_fp,
                             void* thisptr) {
    for (unsigned core = 0; core < sysconf(_SC_NPROCESSORS_ONLN); core++) {
      if (_cpu_mask.check_core(core)) {
        bool ready = false;
        wmb();
        _locks[core].lock();
        /* for each core in the system we need to call the init_fp and save
         * return pointer */
        _work[core].push_back(
            {init_fp, callback_fp, cleanup_fp, nullptr, thisptr, &ready});
        _locks[core].unlock();
        while (!ready) usleep(100);
      }
    }
  }

 private:
  void thread_entry(unsigned core) {
    if (MASK_PROFILING_SIGNAL) {
      sigset_t set;
      sigemptyset(&set);
      sigaddset(&set, SIGPROF);
      int s = pthread_sigmask(SIG_BLOCK, &set, NULL);
      assert(s == 0);
    }

    cpu_mask_t mask;
    mask.add_core(core);
    set_cpu_affinity_mask(mask);

    /* simple round robin */
    while (!_exit_flag) {
      _locks[core].lock();

      for (auto& f : _work[core]) {
        if (_exit_flag) return;
        if (f.arg0 == nullptr) {
          f.arg0 = f.init_function(core, f.thisptr);
          *(f.ready) = true;
          wmb();
        }
        assert(f.callback_function);
        f.callback_function(
            core, f.arg0); /* call registered function with argument */
      }

      _locks[core].unlock();
    }
    /* call cleanup */

    _locks[core].lock();
    for (auto& f : _work[core]) {
      f.cleanup_function(core, f.arg0);
    }
    _locks[core].unlock();
  }

 private:
  struct fp {
    std::function<void*(unsigned, void*)> init_function;
    std::function<void(unsigned, void*)> callback_function;
    std::function<void(unsigned, void*)> cleanup_function;
    void* arg0;
    void* thisptr;
    bool* ready;
  };

  bool _exit_flag = false;
  std::vector<std::thread*> _threads;
  cpu_mask_t _cpu_mask;
  Common::Spin_lock _locks[MAX_CORES];
  std::vector<struct fp> _work[MAX_CORES];
};

}  // namespace Core

#endif

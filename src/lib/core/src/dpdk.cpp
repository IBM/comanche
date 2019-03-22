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

#include "dpdk.h"

#include <rte_eal.h>
#include <rte_eal_memconfig.h>
#include <rte_malloc.h>

extern "C" void spdk_mem_map_init(void);
extern "C" void spdk_vtophys_init(void);

/* hack to fix build */
extern "C" uint32_t rte_net_get_ptype(const struct rte_mbuf *m,
                                      struct rte_net_hdr_lens *hdr_lens,
                                      uint32_t layers) __attribute__((weak));
extern "C" uint32_t rte_net_get_ptype(const struct rte_mbuf *m,
                                      struct rte_net_hdr_lens *hdr_lens,
                                      uint32_t layers) {
  asm("int3");
  return 0;
}

namespace DPDK
{
bool _g_eal_initialized = false;

void eal_init(size_t memory_limit_MB, unsigned master_core, bool primary) {
  std::string proc_type_option =
      primary ? "--proc-type=primary" : "--proc-type=secondary";

  if (!DPDK::_g_eal_initialized) {
    int rc;

    char default_prefix[] = "nvme_comanche";
    char fprefix_[32], wl0_[32], wl1_[32], wl2_[32], wl3_[32];
    char memory_option[32];

    strcpy(memory_option, "");
    fprefix_[0] = wl0_[0] = wl1_[0] = wl2_[0] = wl3_[0] = '\0';

    strcpy(fprefix_, "--file-prefix=");

    char *wl0 = getenv("SPDK_DEVICE0");
    char *mlimit = getenv("SPDK_MEMLIMIT");

    if (wl0) { /* multi-device */
      strcpy(wl0_, "-w ");
      strcat(wl0_, wl0);
      strcat(fprefix_, wl0);

      /* if we specify a device, we're going to bound memory for
         multiple instances */
      if (mlimit) { /* SPDK_MEMLIMIT defined */
        sprintf(memory_option, "-m %s", mlimit);
        PLOG("Using SPDK_MEMLIMIT defined memory limit %s MB", mlimit);
      }
      else {
        if (memory_limit_MB > 0) {
          sprintf(memory_option, "-m %lu", memory_limit_MB);
          PLOG("Using API defined memory limit %lu MB", memory_limit_MB);
        }
        else {
          sprintf(memory_option, "-m %u", CONFIG_MAX_MEMORY_PER_INSTANCE_MB);
          PLOG("Using default multi-instance limit %u MB",
               CONFIG_MAX_MEMORY_PER_INSTANCE_MB);
        }
      }
    }
    else {
      if (mlimit) {
        sprintf(memory_option, "-m %s", mlimit);
        PLOG("Using SPDK_MEMLIMIT defined memory limit %s MB", mlimit);
      }
      else if (memory_limit_MB > 0) {
        sprintf(memory_option, "-m %lu", memory_limit_MB);
        PLOG("Using API defined memory limit %lu MB", memory_limit_MB);
      }
      else {
        PLOG("No memory limit enforced.");
      }

      wl0 = default_prefix;
      strcpy(wl0_, "");
      strcat(fprefix_, default_prefix);
    }

    DEV_OPT_DECL("SPDK_DEVICE1", wl1);
    DEV_OPT_DECL("SPDK_DEVICE2", wl2);
    DEV_OPT_DECL("SPDK_DEVICE3", wl3);

    /* set up core mask */
    int num_cores = numa_num_configured_cpus();
    PLOG("CPU count: %d", num_cores);
    if (num_cores > CONFIG_THREAD_LIMIT) num_cores = CONFIG_THREAD_LIMIT;
    std::stringstream core_mask;
    core_mask << std::hex << num_cores;

    std::string lcores = "0-" + std::to_string(num_cores - 1);

    static const char *ealargs[] = {
        "comanche",
        // "-c",
        // core_mask.str().c_str(),
        "-l",
        lcores.c_str(),  // cores to run on
        "-n",
        "4",  // number of memory channels
        "--master-lcore", std::to_string(master_core).c_str(), memory_option,
        proc_type_option.c_str(), "--log-level=5", fprefix_, wl0_, wl1_, wl2_,
        wl3_};

    /* Save thread affinity. (rte_eal_init will reset it) */
    cpu_set_t cpuset;
    rc = pthread_getaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    assert(rc == 0);

    /*
     * By default, the SPDK NVMe driver uses DPDK for huge page-based
     *  memory management and NVMe request buffer pools.  Huge pages can
     *  be either 2MB or 1GB in size (instead of 4KB) and are pinned in
     *  memory.  Pinned memory is important to ensure DMA operations
     *  never target swapped out memory.
     *
     */
    rc =
        rte_eal_init(sizeof(ealargs) / sizeof(ealargs[0]), (char **) (ealargs));
    if (rc < 0) {
      throw new API_exception("could not initialize DPDK EAL");
    }

    spdk_mem_map_init();
    spdk_vtophys_init();

    PINF("# DPDK EAL set maps & initialized OK.");

    /* Restore thread affinity mask */
    rc = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
    assert(rc == 0);

    DPDK::_g_eal_initialized = true;
    PINF("# DPDK EAL initialized ok (%s).", proc_type_option.c_str());
  }
  else {
    PINF("# DPDK already initialized");
  }
  //  meminfo_display();
}

void eal_show_info(void) { rte_malloc_dump_stats(stderr, NULL); }

void meminfo_display(void) {
  printf("----------- MEMORY_SEGMENTS -----------\n");
  rte_dump_physmem_layout(stdout);
  printf("--------- END_MEMORY_SEGMENTS ---------\n");
}

}  // namespace DPDK

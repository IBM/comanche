/*
 * (C) Copyright IBM Corporation 2017. All rights reserved.
 *
 */

/* 
 * Authors: 
 * 
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#include "eal_init.h"
#include <stdlib.h>
#include <sys/stat.h>
#include <pthread.h>
#include <common/exceptions.h>
#include <rte_eal.h>
#include <rte_eal_memconfig.h>

#include "config.h"

extern "C" void spdk_vtophys_register_dpdk_mem(void);

namespace DPDK {

static bool _g_eal_initialized = false;

#define DEV_OPT_DECL(name, var) char * var = getenv(name);  \
    if(var) {                                               \
      strcpy(var ## _, "-w ");                              \
      strcat(var ## _, var);                                \
    }                                                       \
    else strcpy(var ## _,"");

static void
meminfo_display(void)
{
  printf("----------- MEMORY_SEGMENTS -----------\n");
  rte_dump_physmem_layout(stdout);
  printf("--------- END_MEMORY_SEGMENTS ---------\n");

  // printf("------------ MEMORY_ZONES -------------\n");
  // rte_memzone_dump(stdout);
  // printf("---------- END_MEMORY_ZONES -----------\n");

  // printf("------------- TAIL_QUEUES -------------\n");
  // rte_dump_tailq(stdout);
  // printf("---------- END_TAIL_QUEUES ------------\n");
}


void eal_init(size_t memory_limit_MB)
{
  if (!DPDK::_g_eal_initialized) {
    int rc;

    char default_prefix[] = "nvme_comanche";    
    char fprefix_[32],  wl0_[32], wl1_[32], wl2_[32], wl3_[32];
    char memory_option[32];

    strcpy(memory_option,"");
    fprefix_[0] = wl0_[0] = wl1_[0] = wl2_[0] = wl3_[0] ='\0';

    strcpy(fprefix_,"--file-prefix=");
    
    char * wl0 = getenv("SPDK_DEVICE0");
    char * mlimit = getenv("SPDK_MEMLIMIT");

    if(wl0) { /* multi-device */
      strcpy(wl0_, "-w ");
      strcat(wl0_, wl0);
      strcat(fprefix_,wl0);

      /* if we specify a device, we're going to bound memory for 
         multiple instances */      
      if(mlimit) { /* SPDK_MEMLIMIT defined */
        sprintf(memory_option, "-m %s", mlimit);
        PLOG("Using SPDK_MEMLIMIT defined memory limit %s MB", mlimit);
      }
      else {
        if(memory_limit_MB > 0) {
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
      if(mlimit) {
        sprintf(memory_option, "-m %s", mlimit);
        PLOG("Using SPDK_MEMLIMIT defined memory limit %s MB", mlimit);
      }
      else if(memory_limit_MB > 0) {
        sprintf(memory_option, "-m %lu", memory_limit_MB);
        PLOG("Using API defined memory limit %lu MB", memory_limit_MB);
      }

      else {
        PLOG("No memory limit enforced.");
      }
      wl0 = default_prefix;
      strcpy(wl0_,"");
      strcat(fprefix_,default_prefix);
    }

    DEV_OPT_DECL("SPDK_DEVICE1", wl1);
    DEV_OPT_DECL("SPDK_DEVICE2", wl2);
    DEV_OPT_DECL("SPDK_DEVICE3", wl3);
    
    static const char * ealargs[] = {
      "comanche",
      "-c","0xFFFFF", // core mask
      "-l", "0-8", // cores to run on 
      "-n","2", // number of memory channels
      "--master-lcore", "0",
      memory_option,
      "--proc-type=primary",
      "--log-level=5",
      fprefix_,
      wl0_, wl1_, wl2_, wl3_
    };


    /* Save thread affinity. (rte_eal_init will reset it) */
    cpu_set_t cpuset;
    rc = pthread_getaffinity_np(pthread_self(),sizeof(cpuset),&cpuset);
    assert(rc==0);

    /*
     * By default, the SPDK NVMe driver uses DPDK for huge page-based
     *  memory management and NVMe request buffer pools.  Huge pages can
     *  be either 2MB or 1GB in size (instead of 4KB) and are pinned in
     *  memory.  Pinned memory is important to ensure DMA operations
     *  never target swapped out memory.
     *
     */
    rc = rte_eal_init(sizeof(ealargs) / sizeof(ealargs[0]),
                      (char**)(ealargs));
    if (rc < 0) {
      throw API_exception("could not initialize DPDK EAL");
    }
    PLOG("EAL initialized OK.");


    /* Restore thread affinity mask */
    rc = pthread_setaffinity_np(pthread_self(),sizeof(cpuset),&cpuset);
    assert(rc==0);

    spdk_vtophys_register_dpdk_mem();
    
    DPDK::_g_eal_initialized = true;
  }

  meminfo_display();
}

} // namespace DPDK

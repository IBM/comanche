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

#ifndef __NVME_CONFIG_H__
#define __NVME_CONFIG_H__

#define CONFIG_DEVICE_BLOCK_SIZE (4096)         // block size in bytes - TODO get this from device
#define CONFIG_MAAS_ZERO_NEW_STORAGE           // normally this is on (for Kivati)
#define CONFIG_MAX_MEMORY_PER_INSTANCE_MB 16384 // 16GB size in MB to limit each instance (only for multi-instance)
#define CONFIG_IO_MEMORY_ALIGNMENT_REQUIREMENT 4 // 4 bytes aligned for PRP mode (non-SG) see NVMe spec

//#define CONFIG_FLUSH_CACHES_ON_IO               // includes implicit cache flush for writes; Intel is cache coherent

//#define CONFIG_QUEUE_STATS                     // turn on: statistics
#define CONFIG_STATS_REPORT_INTERVAL 1000000       // interval in IOs to report stats
//#define CHECK_THREAD_VIOLATION                  // turn on/off: thread reentrancy violation checks
//#define CONFIG_CHECKS_VALIDATE_POINTERS // turn on/off extra pointer validity checking
#endif

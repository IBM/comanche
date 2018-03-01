/*
   Copyright [2017] [IBM Corporation]

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

#ifndef __NVME_CONFIG_H__
#define __NVME_CONFIG_H__

#define CONFIG_DEVICE_BLOCK_SIZE (4096)         // block size in bytes - TODO get this from device
#define CONFIG_MAAS_ZERO_NEW_STORAGE           // normally this is on (for Kivati)
#define CONFIG_MAX_MEMORY_PER_INSTANCE_MB 16384 // 16GB size in MB to limit each instance (only for multi-instance)
#define CONFIG_IO_MEMORY_ALIGNMENT_REQUIREMENT 4 // 4 bytes aligned for PRP mode (non-SG) see NVMe spec

//#define CONFIG_QUEUE_STATS                     // turn on: statistics
#undef CONFIG_QUEUE_STATS_DETAILED // enable for detailed queue stats

#ifdef CONFIG_QUEUE_STATS
#define CONFIG_STATS_REPORT_INTERVAL 100000       // interval in IOs to report stats
#endif

//#define CHECK_THREAD_VIOLATION                  // turn on/off: thread reentrancy violation checks
//#define CONFIG_CHECKS_VALIDATE_POINTERS // turn on/off extra pointer validity checking
#endif

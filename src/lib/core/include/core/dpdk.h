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

#ifndef __CORE_DPDK_H__
#define __CORE_DPDK_H__

#include <sstream>
#include <stdlib.h>
#include <sys/stat.h>
#include <pthread.h>
#include <string.h>
#include <numa.h>
#include <common/exceptions.h>

#define CONFIG_MAX_MEMORY_PER_INSTANCE_MB 16384  // 16GB
#define CONFIG_THREAD_LIMIT 48  // limit number of EAL threads

namespace DPDK
{
extern bool _g_eal_initialized;

#define DEV_OPT_DECL(name, var)                                                                                        \
  char* var = getenv(name);                                                                                            \
  if (var) {                                                                                                           \
    strcpy(var##_, "-w ");                                                                                             \
    strcat(var##_, var);                                                                                               \
  }                                                                                                                    \
  else                                                                                                                 \
    strcpy(var##_, "");

void meminfo_display(void);

void eal_init(size_t memory_limit_MB, unsigned master_core = 0, bool primary=true);
void eal_show_info();


}  // namespace DPDK

#endif

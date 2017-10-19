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


/**
 * @file
 * @brief Read TSC (time stamp counter) functions.
 */

#ifndef _RDTSC_H
#define _RDTSC_H

#include <stdint.h>
#include <unistd.h>


/**
 * Read tsc.
 * @return  tsc value.
 */
static inline uint64_t rdtsc(void)
{
    union {
        uint64_t val;
        struct {
            uint32_t lo;
            uint32_t hi;
        };
    } tsc;
    asm volatile ("rdtsc" : "=a" (tsc.lo), "=d" (tsc.hi));
    return tsc.val;
}

/**
 * Get the elapsed tsc since the specified started tsc.
 * @param   tsc         started tsc
 * @return  number of tsc elapsed.
 */
static inline uint64_t rdtsc_elapse(uint64_t tsc) {
    int64_t et;

    do {
        et = rdtsc() - tsc;
    } while (et <= 0);
    return et;
}

/**
 * Get tsc per second using sleeping for 1/100th of a second.
 */
static inline uint64_t rdtsc_second()
{
    static uint64_t tsc_ps = 0;
    if (!tsc_ps) {
        uint64_t tsc = rdtsc();
        usleep(10000);
        tsc_ps = (rdtsc() - tsc) * 100;
    }
    return tsc_ps;
}

#endif // _RDTSC_H


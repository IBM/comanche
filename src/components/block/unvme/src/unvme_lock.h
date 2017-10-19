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
 * @brief UNVMe fast read lock with occasional writes.
 */

#include <sched.h>

/// Lock write bit
#define UNVME_LOCKWBIT      0x80000000

/// Simple read write lock
typedef unsigned unvme_lock_t;


/**
 * Increment read lock and wait if pending write.
 * @param   lock    lock variable
 */
static inline void unvme_lockr(unvme_lock_t* lock)
{
    for (;;) {
        if (!(__sync_fetch_and_add(lock, 1) & UNVME_LOCKWBIT)) return;
        __sync_fetch_and_sub(lock, 1);
        sched_yield();
    }
}

/**
 * Decrement read lock.
 * @param   lock    lock variable
 */
static inline void unvme_unlockr(unvme_lock_t* lock)
{
    __sync_fetch_and_sub(lock, 1);
}

/**
 * Acquire write lock and wait for all pending read/write.
 * @param   lock    lock variable
 */
static inline void unvme_lockw(unvme_lock_t* lock)
{
    for (;;) {
        int val = __sync_fetch_and_or(lock, UNVME_LOCKWBIT);
        if (val == 0) return;
        sched_yield();

        // if not pending write then just wait for all reads to clear
        if (!(val & UNVME_LOCKWBIT)) {
            while (*lock != UNVME_LOCKWBIT) sched_yield();
            return;
        }
    }
}

/**
 * Clear the write lock.
 * @param   lock    lock variable
 */
static inline void unvme_unlockw(unvme_lock_t* lock)
{
    __sync_fetch_and_and(lock, ~UNVME_LOCKWBIT);
}


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
 * @brief Logging header file.
 */

#ifndef _UNVME_LOG_H 
#define _UNVME_LOG_H 

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

/// @cond

#define INFO(fmt, arg...)     log_msg(0, fmt "\n", ##arg)
#define INFO_FN(fmt, arg...)  log_msg(0, "%s " fmt "\n", __func__, ##arg)
#define ERROR(fmt, arg...)    log_msg(1, "ERROR: %s " fmt "\n", __func__, ##arg)

#ifdef UNVME_DEBUG
    #define DEBUG             INFO
    #define DEBUG_FN          INFO_FN
    #define HEX_DUMP          hex_dump
#else
    #define DEBUG(arg...)
    #define DEBUG_FN(arg...)
    #define HEX_DUMP(arg...)
#endif

/// @endcond


// Export function
int log_open(const char* filename, const char* mode);
void log_close();
void log_msg(int err, const char* fmt, ...);


/**
 * Hex dump a data block byte-wise.
 * @param   buf     buffer to read into
 * @param   len     size
 */
static inline void hex_dump(void* buf, int len)
{
    unsigned char* b = buf;
    int i, k = 0, e = 44, t = 44;
    char ss[3906];

    if (len > 1024) len = 1024;

    for (i = 0; i < len; i++) {
        if (!(i & 15)) {
            if (i > 0) {
                ss[k] = ' ';
                ss[t++] = '\n';
                k = t;
            }
            e = t = k + 44;
            k += sprintf(ss+k, "  %04x:", i);
        }
        if (!(i & 3)) ss[k++] = ' ';
        k += sprintf(ss+k, "%02x", b[i]);
        ss[t++] = isprint(b[i]) ? b[i] : '.';
    }
    ss[t] = 0;
    for (i = k; i < e; i++) ss[i] = ' ';
    INFO("%s", ss);
}

/**
 * Invoke calloc and terminated on failure.
 */
static inline void* zalloc(int size)
{
    void* mem = calloc(1, size);
    if (!mem) {
        ERROR("calloc");
        abort();
    }
    return mem;
}

#endif // _UNVME_LOG_H


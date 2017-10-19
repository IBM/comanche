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
 * @brief Logging support routines.
 */

#include <stdarg.h>
#include <pthread.h>

#include "unvme_log.h"


// Static global variables
static FILE*                log_fp = NULL;  ///< log file pointer
static int                  log_count = 0;  ///< log open count
static pthread_mutex_t      log_lock = PTHREAD_MUTEX_INITIALIZER; ///< log lock


/**
 * Open log file.  Only one log file is supported and thus only the first
 * call * will create the log file by its specified name.  Subsequent calls
 * will only be counted.
 * @param   name        log filename
 * @param   mode        open mode
 * @return  0 indicating 
 */
int log_open(const char* name, const char* mode)
{
    pthread_mutex_lock(&log_lock);
    if (!log_fp) {
        log_fp = fopen(name, mode);
        if (!log_fp) {
            perror("log_open");
            pthread_mutex_unlock(&log_lock);
            return -1;
        }
    }
    log_count++;
    pthread_mutex_unlock(&log_lock);
    return 0;
}

/**
 * Close the log file (only the last close will effectively close the file).
 */
void log_close()
{
    pthread_mutex_lock(&log_lock);
    if (log_count > 0) {
        if ((--log_count == 0) && log_fp && log_fp != stdout) {
            fclose(log_fp);
            log_fp = NULL;
        }
    }
    pthread_mutex_unlock(&log_lock);
}

/**
 * Write a formatted message to log file, if log file is opened.
 * If err flag is set then log also to stderr.
 * @param   err         print to stderr indication
 * @param   fmt         formatted message
 */
void log_msg(int err, const char* fmt, ...)
{
    va_list args;

    pthread_mutex_lock(&log_lock);
    if (log_fp) {
        va_start(args, fmt);
        if (err) {
            char s[256];
            vsnprintf(s, sizeof(s), fmt, args);
            fprintf(stderr, "%s", s);
            fprintf(log_fp, "%s", s);
            fflush(log_fp);
        } else {
            vfprintf(log_fp, fmt, args);
            fflush(log_fp);
        }
        va_end(args);
    } else {
        va_start(args, fmt);
        if (err) {
            vfprintf(stderr, fmt, args);
        } else {
            vfprintf(stdout, fmt, args);
            fflush(stdout);
        }
        va_end(args);
    }
    pthread_mutex_unlock(&log_lock);
}


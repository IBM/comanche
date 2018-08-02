#ifndef __KVSTORE_PERF_H__
#define __KVSTORE_PERF_H__

#include <api/kvstore_itf.h>

struct ProgramOptions {
    std::string test;
    std::string component;
    unsigned cores;
    unsigned time_secs;
    std::string path;
    unsigned int size;
    int flags;
    unsigned int elements;
}; 

#endif

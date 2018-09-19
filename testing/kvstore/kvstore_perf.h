#ifndef __KVSTORE_PERF_H__
#define __KVSTORE_PERF_H__

#include <api/kvstore_itf.h>
#include <pthread.h>
#include <sys/stat.h>

#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"

struct ProgramOptions {
    std::string test;
    std::string component;
    unsigned cores;
    unsigned time_secs;
    std::string path;
    unsigned long long int size;
    int flags;
    unsigned int elements;
    Component::IKVStore * store;
    std::string report_file_name;
    unsigned int key_length;
    unsigned int value_length;
    unsigned int bin_count;
    unsigned int bin_threshold_min;
    unsigned int bin_threshold_max;
    int debug_level;
    std::string owner;
    std::string server_address;
    std::string device_name;
}; 

#endif

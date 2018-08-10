#ifndef __KVSTORE_PERF_H__
#define __KVSTORE_PERF_H__

#include <api/kvstore_itf.h>
#include <pthread.h>
#include "rapidjson/document.h"
#include "rapidjson/filestream.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"

struct ProgramOptions {
    std::string test;
    std::string component;
    unsigned cores;
    unsigned time_secs;
    std::string path;
    unsigned int size;
    int flags;
    unsigned int elements;
    Component::IKVStore * store;
    std::string report_file_name;
    unsigned int key_length;
    unsigned int value_length;
}; 

#endif

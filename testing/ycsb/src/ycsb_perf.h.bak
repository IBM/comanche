#ifndef __YCSB_PERF_H__
#define __YCSB_PERF_H__

#include <api/kvstore_itf.h>
#include <pthread.h>
#include <sys/stat.h>
#include "properties.h"

#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"

#define DEFAULT_COMPONENT "filestore"
#define PMSTORE_PATH "libcomanche-pmstore.so"
#define FILESTORE_PATH "libcomanche-storefile.so"
#define NVMESTORE_PATH "libcomanche-nvmestore.so"
#define ROCKSTORE_PATH "libcomanche-rocksdb.so"
#define DAWN_PATH "libcomanche-dawn-client.so"

extern boost::program_options::options_description g_desc;

void show_program_options();
ycsbutils::Properties props;

#endif

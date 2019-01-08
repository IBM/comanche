#include "exp_throughput.h"

std::mutex ExperimentThroughput::_iops_lock;
unsigned long ExperimentThroughput::_iops;

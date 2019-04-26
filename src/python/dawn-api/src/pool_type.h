#ifndef __POOL_TYPE_H__
#define __POOL_TYPE_H__

#include <api/dawn_itf.h>
#include <api/kvstore_itf.h>

typedef struct {
  PyObject_HEAD
  Component::IDawn *          _dawn;
  Component::IKVStore::pool_t _pool;
} Pool;

Pool * Pool_new();

#endif


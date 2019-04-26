/*
 * (C) Copyright IBM Corporation 2018. All rights reserved.
 *
 */

/* 
 * Authors: 
 * 
 * Feng Li (fengggli@yahoo.com)
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 */

#ifndef STATE_MAP_H_
#define STATE_MAP_H_

#include <unordered_map>
#include <pthread.h>
#include <common/rwlock.h>
#include <tbb/concurrent_hash_map.h>
#include <api/kvstore_itf.h>

using pool_t     = uint64_t;

struct volatile_state_t
{
  Common::RWLock _lock;
};

typedef std::unordered_map<const void*, volatile_state_t> pool_state_map_t;
typedef tbb::concurrent_hash_map<const pool_t, pool_state_map_t> state_map_t;

/* we use a concurrent/thread-safe map so we can support multiple
   threads on different pools
   TODO: cleaning up out pool entries? */

class State_map
{
public:
  bool state_get_read_lock(const pool_t pool, const void * ptr);
  bool state_get_write_lock(const pool_t pool, const void * ptr);
  void state_unlock(const pool_t pool, const void * ptr);
  void state_remove(const pool_t pool, const void * ptr);
private:
  state_map_t _state_map;
};


#endif

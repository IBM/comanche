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
#include "state_map.h"

bool State_map::state_get_read_lock(const pool_t pool, const void * ptr) {
  state_map_t::accessor a;
  _state_map.insert(a, pool);
  auto& pool_state_map = a->second;
  auto entry = pool_state_map.find(ptr);
  if(entry == pool_state_map.end())  /* create new entry */
    return pool_state_map[ptr]._lock.read_lock() == 0;
  else 
    return entry->second._lock.read_trylock() == 0;
}

bool State_map::state_get_write_lock(const pool_t pool, const void * ptr) {
  state_map_t::accessor a;
  _state_map.insert(a, pool);
  auto& pool_state_map = a->second;
  auto entry = pool_state_map.find(ptr);
  if(entry == pool_state_map.end())  /* create new entry */
    return pool_state_map[ptr]._lock.write_lock() == 0;
  else 
    return entry->second._lock.write_trylock() == 0;
}

void State_map::state_unlock(const pool_t pool, const void * ptr) {
  state_map_t::accessor a;
  _state_map.insert(a, pool);
  auto& pool_state_map = a->second;
  auto entry = pool_state_map.find(ptr);
  if(entry == pool_state_map.end() || entry->second._lock.unlock())
    throw General_exception("invalid unlock");
}

void State_map::state_remove(const pool_t pool, const void * ptr) {
  state_map_t::accessor a;
  _state_map.insert(a, pool);
  auto& pool_state_map = a->second;
  auto entry = pool_state_map.find(ptr);
  if(entry == pool_state_map.end())
    throw General_exception("invalid unlock");
  pool_state_map.erase(entry);
}


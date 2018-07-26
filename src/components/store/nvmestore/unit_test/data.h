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

#ifndef __DATA_H__
#define __DATA_H__

#include <common/str_utils.h>
#include <common/utils.h>
#include <cstring>
#include <common/logging.h>
#include <common/exceptions.h>

class Data
{
public:
  static constexpr size_t NUM_ELEMENTS = 10000; 
  static constexpr size_t KEY_LEN = 8;
  static constexpr size_t VAL_LEN = KB(128);

  struct KV_pair {
    char key[KEY_LEN + 1];
    char value[VAL_LEN + 1];
  };

  KV_pair * _data;

  Data() {
    PLOG("Initializing data....");
    _data = new KV_pair[NUM_ELEMENTS];

    for(size_t i=0;i<NUM_ELEMENTS;i++) {
      auto key = Common::random_string(KEY_LEN);
      auto val = Common::random_string(VAL_LEN);
      strncpy(_data[i].key, key.c_str(), key.length());
      _data[i].key[KEY_LEN] = '\0';
      strncpy(_data[i].value, val.c_str(), val.length());
     _data[i].value[VAL_LEN] = '\0';
    }
    PLOG("..OK.");
  }

  ~Data() {
    delete [] _data;
  }

  const char * key(size_t i) const {
    if(i >= NUM_ELEMENTS) throw General_exception("out of bounds");
    return _data[i].key;
  }

  const char * value(size_t i) const {
    if(i >= NUM_ELEMENTS) throw General_exception("out of bounds");
    return _data[i].value;
  }

  size_t value_len() const { return VAL_LEN; }

  size_t num_elements() const { return NUM_ELEMENTS; }
};

#endif

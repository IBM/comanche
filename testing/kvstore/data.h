#ifndef __DATA_H__
#define __DATA_H__

#include <sys/mman.h>
#include <boost/program_options.hpp>
#include <common/str_utils.h>
#include <common/logging.h>
#include <common/exceptions.h>


class KV_pair 
{
public:
  std::string key;
  std::string value;
};

class Data
{
public:

  size_t _num_elements;
  size_t _key_len;  
  size_t _val_len; 

  KV_pair * _data = nullptr;
  
  Data() 
  {
    initialize_data(); 
  }
 
  Data(size_t num_elements)
  {
    _num_elements = num_elements;

    initialize_data();
  }

  Data(size_t num_elements, size_t key_len, size_t val_len, bool delay_initialization=false)
  {
    _num_elements = num_elements;
    _key_len = key_len;
    _val_len = val_len;

    if (!delay_initialization)
      {
        initialize_data();
      }
  }

  ~Data() 
  {
    delete [] _data;
  }

  void initialize_data(bool allocate_memory=true)
  {
    PLOG("Initializing data: %d key length, %d value length, %d elements....",
         (int)_key_len, (int)_val_len, (int)_num_elements);
        
    if (allocate_memory)
      {
        _data = new KV_pair[_num_elements];
      }
       
    
    for(size_t i=0;i<_num_elements;i++) 
      {
        auto key = Common::random_string(_key_len);
        auto val = Common::random_string(_val_len);

        _data[i].key = key;
        _data[i].value = val;
      }

    PLOG("%d elements initialized, size %d.", (int)_num_elements, (int)_val_len);
  }
  
  const char * key(size_t i) const 
  {
    if(i >= _num_elements) throw General_exception("index out of bounds");
    return _data[i].key.c_str();
  }
  
  const char * value(size_t i) const 
  {
    if(i >= _num_elements) throw General_exception("index out of bounds");
    return _data[i].value.c_str();
  }
  
  size_t value_len() const { return _val_len; }
  
  size_t num_elements() const { return _num_elements; }
};

#endif

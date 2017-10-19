#ifndef __COMANCHE_LOGICAL_VOLUME_H__
#define __COMANCHE_LOGICAL_VOLUME_H__

#include <string>
#include "policy.h"

class Logical_volume
{
  static constexpr uint32_t MAGIC = 0xF1EE1010;

public:

  Logical_volume(const char * volume_name);

  bool sane();

  void add_policy(Policy * p);

  inline Policy * get_policy() const {  return _policy;  }
  const char * name() const { return (const char *) _name.c_str(); }  
  size_t block_size() const { return 4096;/* to FIX */ }

private:
  uint32_t    _magic; // sanity check
  std::string _name;
  Policy *    _policy;
};


#endif //__COMANCHE_LOGICAL_VOLUME_H__

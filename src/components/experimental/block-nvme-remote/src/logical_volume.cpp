#include "logical_volume.h"

Logical_volume::Logical_volume(const char * volume_name) :
  _magic(MAGIC),
  _name(volume_name),
  _policy(nullptr)
{  
}


bool Logical_volume::sane() {
  return _magic == MAGIC;
}


void Logical_volume::add_policy(Policy * p) {
  assert(_magic==MAGIC);
  if(_policy)
    throw General_exception("policy chaining not supported");
  
  _policy = p;
}


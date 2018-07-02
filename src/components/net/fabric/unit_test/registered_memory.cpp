#include "registered_memory.h"

registered_memory::registered_memory(Component::IFabric_connection &cnxn_)
  : _cnxn(cnxn_)
  , _memory{}
  , _registration(_cnxn, &*_memory.begin(), _memory.size(), remote_key, 0U)
{}

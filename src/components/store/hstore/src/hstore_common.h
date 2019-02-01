#ifndef COMANCHE_HSTORE_COMMON_H
#define COMANCHE_HSTORE_COMMON_H

#include <string>

#define PREFIX "HSTORE : %s: "

using IKVStore = Component::IKVStore;

namespace
{
  std::string make_full_path(const std::string &prefix, const std::string &suffix)
  {
    return prefix + ( prefix[prefix.length()-1] != '/' ? "/" : "") + suffix;
  }
}

#endif

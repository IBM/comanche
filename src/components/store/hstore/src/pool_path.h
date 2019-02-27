/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#ifndef COMANCHE_HSTORE_POOL_PATH_H
#define COMANCHE_HSTORE_POOL_PATH_H

#include <string>

class pool_path
{
  std::string _name;

public:
  explicit pool_path(
    const std::string &name_
  )
    : _name(name_)
  {}

  pool_path()
    : pool_path("")
  {}

  std::string str() const
  {
    return _name;
  }
};

#endif

/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#ifndef COMANCHE_HSTORE_POOL_PATH_H
#define COMANCHE_HSTORE_POOL_PATH_H

#include <string>

class pool_path
{
  std::string _dir;
  std::string _name;

public:
  explicit pool_path(
    const std::string &dir_
    , const std::string &name_
  )
    : _dir(dir_)
    , _name(name_)
  {}

  pool_path()
    : pool_path("/", "")
  {}

  std::string str() const
  {
    /* ERROR: fails if _dir is the empty string */
    return _dir + ( _dir[_dir.length()-1] != '/' ? "/" : "") + _name;
  }
#if 1
  /* delete_pool only */
  const std::string &dir() const noexcept { return _dir; }
  const std::string &name() const noexcept { return _name; }
#endif
};

#endif

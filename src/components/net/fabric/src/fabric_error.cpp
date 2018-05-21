/*
   Copyright [2018] [IBM Corporation]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include "fabric_error.h"

/*
 * Authors:
 *
 */

#include <rdma/fi_errno.h>

#include <string>
#include <stdexcept>

/**
 * Fabric/RDMA-based network component
 */

fabric_error::fabric_error(int i_, const char *file_, int line_)
  : std::logic_error{std::string{"fabric_error \""} + fi_strerror(i_) + "\" at " + file_ + ":" + std::to_string(line_)}
  , _i(i_)
  , _file(file_)
  , _line(line_)
{}

fabric_error::fabric_error(int i_, const char *file_, int line_, const std::string &desc_)
  : std::logic_error{std::string{"fabric_error ("} + std::to_string(i_) + ") \"" + fi_strerror(i_) + "\" at " + file_ + ":" + std::to_string(line_) + " " + desc_}
  , _i(i_)
  , _file(file_)
  , _line(line_)
{}

fabric_error fabric_error::add(const std::string &added) const
{
  return fabric_error(_i, _file, _line, added);
}

fabric_bad_alloc::  fabric_bad_alloc(std::string which)
  : std::bad_alloc{}
  , _what{"fabric_bad_alloc " + which}
{}

const char *fabric_bad_alloc::what() const noexcept
{
  return _what.c_str();
}

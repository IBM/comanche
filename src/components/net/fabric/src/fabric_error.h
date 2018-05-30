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

#ifndef _FABRIC_ERROR_H_
#define _FABRIC_ERROR_H_

/*
 * Authors:
 *
 */

#include <string>
#include <stdexcept>

/**
 * Fabric/RDMA-based network component
 */

/* A simplification; most fabric errors will not be logic errors */
class fabric_error
  : public std::logic_error
{
  int _i;
  const char *_file;
  int _line;
public:
  fabric_error(int i, const char *file, int line);
  fabric_error(int i, const char *file, int line, const std::string &desc);
  fabric_error(const fabric_error &) = default;
  fabric_error& operator=(const fabric_error &) = default;
  fabric_error add(const std::string &added) const;
  unsigned id() const { return unsigned(_i); }
};

class fabric_bad_alloc
  : public std::bad_alloc
{
  std::string _what;
public:
  fabric_bad_alloc(std::string which);
  const char *what() const noexcept override;
};

#endif

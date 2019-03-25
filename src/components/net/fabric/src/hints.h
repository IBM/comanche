/*
   Copyright [2017-2019] [IBM Corporation]
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


/*
 * Authors:
 *
 */

#ifndef _FABRIC_HINTS_H_
#define _FABRIC_HINTS_H_

#include <cstdint> /* uint64_t */
#include <memory> /* shared_ptr */

struct fi_info;

/**
 * Fabric/RDMA-based network component
 *
 */

class hints
{
  std::shared_ptr<fi_info> _info;
public:
  explicit hints();
  /**
   * @throw std::bad_alloc : fabric_bad_alloc - out of memory
   */
  explicit hints(std::shared_ptr<fi_info> info);
  hints &caps(std::uint64_t c);
  hints &mode(std::uint64_t c);
  hints &mr_mode(int m);
  hints &prov_name(const char *n);
  const char *prov_name() const;
  const fi_info &data();
};

#endif

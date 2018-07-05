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

#ifndef _FABRIC_STR_H_
#define _FABRIC_STR_H_

/*
 * Authors:
 *
 */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#include <rdma/fabric.h> /* fi_tostr */
#pragma GCC diagnostic pop

#include <mutex>
#include <string>

struct fi_info;
struct fi_fabric_attr;

/**
 * Fabric/RDMA-based network component
 *
 */

/* bind fi types to their type ids */
template <typename T>
  struct str_attr;

template <>
  struct str_attr<::fi_info>
  {
    static constexpr auto id = FI_TYPE_INFO;
  };

template <>
  struct str_attr<::fi_fabric_attr>
  {
    static constexpr auto id = FI_TYPE_FABRIC_ATTR;
  };

extern std::mutex m_fi_tostr;

/* typesafe and threadsafe version of fi_tostr */
template <typename T>
  std::string tostr(const T &f)
  {
    std::lock_guard<std::mutex> g{m_fi_tostr};
    return ::fi_tostr(&f, str_attr<T>::id);
  }

#endif

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


#ifndef _FABRIC_UTIL_H_
#define _FABRIC_UTIL_H_

/*
 * Authors:
 *
 */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#include <rdma/fabric.h> /* fid_t */
#pragma GCC diagnostic pop

#include "fabric_types.h" /* addr_ep_t */

#include <cstdint> /* uint32_t */
#include <memory> /* shared_ptr */
#include <string>

struct fi_info;

/**
 * Fabric/RDMA-based network component
 *
 */

/**
 * @throw std::bad_alloc : fabric_bad_alloc - libfabric out of memory
 */
std::shared_ptr<::fi_info> make_fi_info();

/**
 * @throw std::bad_alloc : fabric_bad_alloc - libfabric out of memory
 */
std::shared_ptr<::fi_info> make_fi_infodup(const ::fi_info &info_, const std::string &why_);

/**
 * @throw fabric_runtime_error : std::runtime_error : ::fi_getname fail
 */
auto get_name(::fid_t fid) -> fabric_types::addr_ep_t;

const char *get_event_name(std::uint32_t e);

#endif

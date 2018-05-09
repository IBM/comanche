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

#ifndef _ENDPOINT_RESOURCES_SIMPLEX_H_
#define _ENDPOINT_RESOURCES_SIMPLEX_H_

#include <rdma/fi_domain.h>

#include "allocator_aligned.h"
#include "fabric_ptr.h" /* fid_unique_ptr */

#include <cstdint> /* uint64_t */
#include <vector>

struct fid_cq;
struct fid_domain;
struct fid_ep;
struct fid_mr;

class Fabric_endpoint_active;

class endpoint_resources_simplex
{
public:
  using buffer_t = std::vector<char, aligned_allocator<char>>;
private:
  buffer_t _buffer;
  fid_unique_ptr<fid_mr> _mr;
  fi_cq_attr _cq_attr;
  fid_unique_ptr<fid_cq> _cq;
  std::uint64_t _seq;
  std::uint64_t _cq_ctr;
  void *_mr_desc;
public:
  explicit endpoint_resources_simplex(fid_domain &d, std::uint64_t access, std::uint64_t key, std::uint64_t flags);

  fid_t cq() const { return &_cq->fid; }
  std::uint64_t seq() const { return _seq; }
  void get_cq_comp(uint64_t limit);
  buffer_t &buffer() { return _buffer; }
  void *mr_desc() const { return _mr_desc; }
};

#endif

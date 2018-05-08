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

#include "endpoint_resources_simplex.h"

#include "fabric_error.h"

#include "fabric_help.h"

#include <stdexcept>

#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>

endpoint_resources_simplex::endpoint_resources_simplex(fid_domain &d, std::uint64_t access_, uint64_t key_, uint64_t flags_
)
  : _buffer(10000U) /* ERROR: 10K is probably *not* good enough for everybody */
  , _mr(make_fid_mr_reg(d, &*_buffer.begin(), _buffer.size(), access_, key_, flags_))
  , _cq_attr{}
  , _cq(make_fid_cq(d, _cq_attr, this))
  , _seq(0u)
  , _cq_ctr(0u)
  , _mr_desc(fi_mr_desc(&*_mr))
{
}

class cq_error
  : public std::runtime_error
{
  fi_cq_err_entry _err;
public:
  cq_error(fid_cq &cq_, fi_cq_err_entry &&e_)
    : std::runtime_error(fi_cq_strerror(&cq_, e_.prov_errno, e_.err_data, nullptr, 0))
    , _err(std::move(e_))
  {}
};

[[noreturn]] void get_cq_comp_err(fid_cq &cq_, uint64_t &cur_)
{
  fi_cq_err_entry err;

  auto r = fi_cq_readerr(&cq_, &err, 0);
  if ( r < 0 )
  {
    throw fabric_error(-r, __FILE__, __LINE__);
  }
  ++cur_;
  throw cq_error(cq_, std::move(err));
}


static void get_cq_comp(fid_cq &cq_, uint64_t &cur_, uint64_t limit_)
{
  while ( cur_ < limit_ )
  {
    fi_cq_err_entry comp;
    switch ( auto r = fi_cq_read(&cq_, &comp, 1) )
    {
    case -FI_EAVAIL:
      get_cq_comp_err(cq_, cur_);
      break;
    case -FI_EAGAIN:
      break;
    default:
      if ( r < 0 )
      {
        throw fabric_error(-r, __FILE__, __LINE__);
      }
      ++cur_;
      break;
    }
  }
}

void endpoint_resources_simplex::get_cq_comp(uint64_t limit)
{
  return ::get_cq_comp(*this->_cq, this->_cq_ctr, limit);
}

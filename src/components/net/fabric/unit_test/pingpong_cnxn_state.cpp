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
#include "pingpong_cnxn_state.h"

#include <api/fabric_itf.h>
#include "pingpong_buffer_state.h"

cnxn_state::cnxn_state(
  Component::IFabric_active_endpoint_comm &comm_
  , unsigned iteration_count_
  , std::size_t msg_size_
)
  : _comm(&comm_)
  , _max_inject_size{_comm->max_inject_size()}
  , msg_size{msg_size_}
  , iterations_left(iteration_count_)
  , done(false)
{
}

void cnxn_state::send(buffer_state &bt_, cb_ctxt *tx_ctxt_)
{
  if ( bt_.v.size() == 1 && bt_.v.front().iov_len <= _max_inject_size )
  {
    comm().inject_send(bt_.v.front().iov_base, bt_.v.front().iov_len);
    --iterations_left;
  }
  else
  {
    comm().post_send(&*bt_.v.begin(), &*bt_.v.end(), &*bt_.d.begin(), tx_ctxt_);
  }
}

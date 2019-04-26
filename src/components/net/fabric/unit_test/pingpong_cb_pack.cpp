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
#include "pingpong_cb_pack.h"

cb_pack::cb_pack(
  cnxn_state &state_
  , Component::IFabric_connection &cnxn_
  , cb_ctxt::cb_t send_cb
  , cb_ctxt::cb_t recv_cb
  , std::uint64_t buffer_size_
  , std::uint64_t remote_key_base_
  , std::uint64_t msg_size_
)
  : _tx_ctxt(state_, send_cb, nullptr, cnxn_, buffer_size_, remote_key_base_*3U, msg_size_, "tx")
  , _rx0_ctxt(state_, recv_cb, &_tx_ctxt, cnxn_, buffer_size_, remote_key_base_*3U+1, msg_size_, "rx0")
  , _rx1_ctxt(state_, recv_cb, &_tx_ctxt, cnxn_, buffer_size_, remote_key_base_*3U+2, msg_size_, "rx1")
{
}

cb_pack::~cb_pack()
{
}

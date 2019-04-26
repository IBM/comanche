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
#include "pingpong_server_cb.h"

#include "pingpong_cb_ctxt.h"
#include "pingpong_server_client_state.h"

#include <gtest/gtest.h>

void pingpong_server_cb::recv_cb(cb_ctxt *rx_ctxt_, ::status_t stat_, std::size_t len_)
{
  auto &cs = rx_ctxt_->state();
  auto &br = rx_ctxt_->buffer();
  EXPECT_EQ(stat_, S_OK);
  EXPECT_EQ(len_, cs.msg_size);
  EXPECT_EQ(br.v.size(), 1U);

  auto tx_ctxt = rx_ctxt_->response();
  auto &bt = tx_ctxt->buffer();
  static_cast<uint8_t *>(bt.v.front().iov_base)[0] = static_cast<uint8_t *>(br.v.front().iov_base)[0];
  static_cast<uint8_t *>(bt.v.front().iov_base)[1] = static_cast<uint8_t *>(br.v.front().iov_base)[1];
  cs.comm().post_recv(&*br.v.begin(), &*br.v.end(), &*br.d.begin(), rx_ctxt_);
  cs.send(bt, tx_ctxt);
  if ( cs.iterations_left == 0 )
  {
    cs.done = true;
  }
}

void pingpong_server_cb::send_cb(cb_ctxt *, ::status_t stat_, std::size_t)
{
  EXPECT_EQ(stat_, S_OK);
}

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

#include "pingpong_server_client_state.h"

#include <common/errors.h> /* S_OK */
#include <common/types.h> /* status_t */
#include <gtest/gtest.h>

#include <cstddef> /* size_t */
#include <cstdint> /* uint64_t */
#include <utility> /* move */

namespace
{
  auto recv_cb(cnxn_state *cs_, buffer_state &br, ::status_t stat_, std::size_t len_, cb_ctxt *ctxt) -> void
  {
    EXPECT_EQ(stat_, S_OK);
    EXPECT_EQ(len_, cs_->msg_size);
    EXPECT_EQ(br.v.size(), 1U);
    cs_->_comm.post_send(&*cs_->bt.v.begin(), &*cs_->bt.v.end(), &*cs_->bt.d.begin(), &cs_->send_ctxt);
    cs_->_comm.post_recv(&*br.v.begin(), &*br.v.end(), &*br.d.begin(), ctxt);
  }
  auto recv0_cb(cnxn_state *cs_, ::status_t stat_, std::uint64_t, std::size_t len_, void *) -> void
  {
    recv_cb(cs_, cs_->br[0], stat_, len_, &cs_->recv0_ctxt);
  }
  auto recv1_cb(cnxn_state *cs_, ::status_t stat_, std::uint64_t, std::size_t len_, void *) -> void
  {
    recv_cb(cs_, cs_->br[1], stat_, len_, &cs_->recv1_ctxt);
  }
  auto send_cb(cnxn_state *cs_, ::status_t stat_, std::uint64_t, std::size_t, void *) -> void
  {
    EXPECT_EQ(stat_, S_OK);
    EXPECT_EQ(cs_->bt.v.size(), 1U);
    --cs_->iterations_left;
  }
}

client_state::client_state(
  Component::IFabric_server_factory &factory_
  , std::size_t buffer_size_
  , std::uint64_t remote_key_
  , unsigned iteration_count_
  , std::size_t msg_size_
)
  : sc(factory_)
  , st(send_cb, recv0_cb, recv1_cb, sc.cnxn(), buffer_size_, remote_key_, iteration_count_, msg_size_)
{
}

client_state::client_state(client_state &&cs_)
  : sc(std::move(cs_.sc))
  , st(std::move(cs_.st))
{
}

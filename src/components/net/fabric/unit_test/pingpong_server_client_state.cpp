#include "pingpong_server_client_state.h"

#include <common/errors.h> /* S_OK */
#include <gtest/gtest.h>

#include <cstddef> /* size_t */
#include <cstdint> /* uint64_t */
#include <utility> /* move */

namespace Component
{
  class IFabric_server_factory;
}

namespace
{
  auto recv_cb(client_state *cs_, buffer_state &br, ::status_t stat_, std::size_t len_, cb_ctxt *ctxt) -> void
  {
    EXPECT_EQ(stat_, S_OK);
    EXPECT_EQ(len_, cs_->msg_size);
    EXPECT_EQ(br.v.size(), 1U);
    cs_->sc.cnxn().post_send(&*cs_->bt.v.begin(), &*cs_->bt.v.end(), &*cs_->bt.d.begin(), &cs_->send_ctxt);
    cs_->sc.cnxn().post_recv(&*br.v.begin(), &*br.v.end(), &*br.d.begin(), ctxt);
  }
  auto recv0_cb(client_state *cs_, ::status_t stat_, std::uint64_t, std::size_t len_, void *) -> void
  {
    recv_cb(cs_, cs_->br[0], stat_, len_, &cs_->recv0_ctxt);
  }
  auto recv1_cb(client_state *cs_, ::status_t stat_, std::uint64_t, std::size_t len_, void *) -> void
  {
    recv_cb(cs_, cs_->br[1], stat_, len_, &cs_->recv1_ctxt);
  }
  auto send_cb(client_state *cs_, ::status_t stat_, std::uint64_t, std::size_t, void *) -> void
  {
    EXPECT_EQ(stat_, S_OK);
    EXPECT_EQ(cs_->bt.v.size(), 1U);
    /* We got a callback on the final send. Expect no more from this client */
    --cs_->iterations_left;
  }
}

void cb_ctxt::cb(void *ctxt_, ::status_t stat_, std::uint64_t flags_, std::size_t len_, void *err_)
{
  auto ctxt = static_cast<const cb_ctxt *>(ctxt_);
  /* calls one of send_cb, recv0_cb, recv1_cb */
  return (ctxt->cb_call)(ctxt->state, stat_, flags_, len_, err_);
}

cb_ctxt::cb_ctxt(client_state *state_, cb_t cb_call_)
  : state(state_)
  , cb_call(cb_call_)
{
}

buffer_state::buffer_state(Component::IFabric_connection &cnxn_, std::size_t buffer_size_, std::uint64_t remote_key_, std::size_t msg_size_)
  : _rm{cnxn_, buffer_size_,  remote_key_}
  , v{{&_rm[0], msg_size_}}
  , d{{_rm.desc()}}
{}

client_state::client_state(
  Component::IFabric_server_factory &factory_
  , std::size_t buffer_size_
  , std::uint64_t remote_key_
  , unsigned iteration_count_
  , std::size_t msg_size_
)
  : send_ctxt(this, &send_cb)
  , recv0_ctxt(this, &recv0_cb)
  , recv1_ctxt(this, &recv1_cb)
  , sc(factory_)
  , msg_size{msg_size_}
  , br{
    }
  , bt{
      sc.cnxn(), buffer_size_, remote_key_*3U+2, msg_size
    }
  , iterations_left(iteration_count_)
{
  for ( auto i = 0U; i != 2U; ++i )
  {
    br.emplace_back(sc.cnxn(), buffer_size_, remote_key_*3U+1, msg_size);
  }
}

client_state::client_state(client_state &&cs_)
  : send_ctxt(this, cs_.send_ctxt.cb_call)
  , recv0_ctxt(this, cs_.recv0_ctxt.cb_call)
  , recv1_ctxt(this, cs_.recv1_ctxt.cb_call)
  , sc(std::move(cs_.sc))
  , msg_size(std::move(cs_.msg_size))
  , br(std::move(cs_.br))
  , bt(std::move(cs_.bt))
  , iterations_left(std::move(cs_.iterations_left))
{
}

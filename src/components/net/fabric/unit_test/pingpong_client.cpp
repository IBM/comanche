#include "pingpong_client.h"

#include "patience.h" /* open_connection_patiently */
#include "pingpong_cnxn_state.h"

#include <common/errors.h> /* S_OK */
#include <common/types.h> /* status_t */
#include <gtest/gtest.h>
#include <exception>
#include <iostream> /* cerr */

namespace Component
{
  class IFabric;
}

namespace
{
  auto recv_cb(cnxn_state *cs_, buffer_state &br, ::status_t stat_, std::size_t len_, cb_ctxt *ctxt) -> void
  {
    EXPECT_EQ(stat_, S_OK);
    EXPECT_EQ(len_, cs_->msg_size);
    EXPECT_EQ(br.v.size(), 1U);
    --cs_->iterations_left;
    if ( cs_->iterations_left != 0 )
    {
      cs_->_comm.post_send(&*cs_->bt.v.begin(), &*cs_->bt.v.end(), &*cs_->bt.d.begin(), &cs_->send_ctxt);
      cs_->_comm.post_recv(&*br.v.begin(), &*br.v.end(), &*br.d.begin(), ctxt);
    }
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
  }
}

pingpong_client::pingpong_client(
  Component::IFabric &fabric_
  , const std::string &fabric_spec_
  , const std::string ip_address_
  , std::uint16_t port_
  , std::uint64_t buffer_size_
  , std::uint64_t remote_key_base_
  , unsigned iteration_count_
  , std::uint64_t msg_size_
)
try
  : _cnxn(open_connection_patiently(fabric_, fabric_spec_, ip_address_, port_))
  , _stat{}
{
  cnxn_state c(send_cb, recv0_cb, recv1_cb, *_cnxn, buffer_size_, remote_key_base_, iteration_count_, msg_size_);

  std::uint64_t poll_count = 0U;
  c._comm.post_recv(&*c.br[0].v.begin(), &*c.br[0].v.end(), &*c.br[0].d.begin(), &c.recv0_ctxt);
  c._comm.post_recv(&*c.br[1].v.begin(), &*c.br[1].v.end(), &*c.br[1].d.begin(), &c.recv1_ctxt);

  _stat.do_start();

  c._comm.post_send(&*c.bt.v.begin(), &*c.bt.v.end(), &*c.bt.d.begin(), &c.send_ctxt);
  while (  c.iterations_left != 0U )
  {
    ++poll_count;
    c._comm.poll_completions(cb_ctxt::cb);
  }

  _stat.do_stop(poll_count);
}
catch ( std::exception &e )
{
  std::cerr << __func__ << ": " << e.what() << "\n";
  throw;
}

pingpong_stat pingpong_client::time() const
{
  return _stat;
}

pingpong_client::~pingpong_client()
{
}

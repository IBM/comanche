#include "pingpong_server_n.h"

#include "eyecatcher.h"
#include <api/fabric_itf.h> /* IFabric_server_factory */
#include <common/errors.h> /* S_OK */
#include <gtest/gtest.h>

#include <exception>
#include <functional> /* ref */
#include <iostream> /* cerr */

namespace
{
  auto cb(void *ctxt_, ::status_t stat_, std::uint64_t flags_, std::size_t len_, void *err_) -> void
  {
    auto ctxt = static_cast<const cb_ctxt *>(ctxt_);
    /* calls one of recv_cb or send_cb */
    return (ctxt->cb_call)(ctxt->state, stat_, flags_, len_, err_);
  }
  auto recv_cb(client_state *cs_, ::status_t stat_, std::uint64_t, std::size_t len_, void *) -> void
  {
    EXPECT_EQ(stat_, S_OK);
    EXPECT_EQ(len_, cs_->msg_size);
    EXPECT_EQ(cs_->v.size(), 1U);
    cs_->sc.cnxn().post_send(&*cs_->v.begin(), &*cs_->v.end(), &*cs_->d.begin(), &cs_->send_ctxt);
  }
  auto send_cb(client_state *cs_, ::status_t stat_, std::uint64_t, std::size_t, void *) -> void
  {
    EXPECT_EQ(stat_, S_OK);
    EXPECT_EQ(cs_->v.size(), 1U);
    /* We got a callback on the final send. Expect no more from this client */
    --cs_->iterations_left;
    if ( cs_->iterations_left )
    {
      cs_->sc.cnxn().post_recv(&*cs_->v.begin(), &*cs_->v.end(), &*cs_->d.begin(), &cs_->recv_ctxt);
    }
  }
}

cb_ctxt::cb_ctxt(client_state *state_, cb_t cb_call_)
  : state(state_)
  , cb_call(cb_call_)
{
}

client_state::client_state(
  Component::IFabric_server_factory &factory_
  , std::size_t buffer_size_
  , std::uint64_t remote_key_
  , unsigned iteration_count_
  , std::size_t msg_size_
)
  : send_ctxt(this, &send_cb)
  , recv_ctxt(this, &recv_cb)
  , sc(factory_)
  , rm{sc.cnxn(), buffer_size_, remote_key_}
  , v{{&rm[0], msg_size_}}
  , d{{rm.desc()}}
  , msg_size{msg_size_}
  , iterations_left(iteration_count_)
{
}

client_state::client_state(client_state &&cs_)
  : send_ctxt(this, cs_.send_ctxt.cb_call)
  , recv_ctxt(this, cs_.recv_ctxt.cb_call)
  , sc(std::move(cs_.sc))
  , rm(std::move(cs_.rm))
  , v(std::move(cs_.v))
  , d(std::move(cs_.d))
  , msg_size(std::move(cs_.msg_size))
  , iterations_left(std::move(cs_.iterations_left))
{
}

void pingpong_server_n::listener(
  std::size_t /* msg_size_ */
)
try
{
  for ( auto &c : _cs )
  {
    c.sc.cnxn().post_recv(&*c.v.begin(), &*c.v.end(), &*c.d.begin(), &c.recv_ctxt);
  }

  std::uint64_t poll_count = 0U;
  auto polled_any = true;
  while ( polled_any )
  {
    polled_any = false;
    for ( auto &c : _cs )
    {
      if ( c.iterations_left != 0 )
      {
        if ( _stat.start() == std::chrono::high_resolution_clock::time_point::min() )
        {
          _stat.do_start();
        }
        c.sc.cnxn().poll_completions(cb);
        ++poll_count;
        polled_any = true;
      }
    }
  }

  _stat.do_stop(poll_count);
}
catch ( std::exception &e )
{
  std::cerr << "pingpong_server::" << __func__ << ": " << e.what() << "\n";
  throw;
}

namespace
{
  std::vector<client_state> clients(unsigned count_, Component::IFabric_server_factory &factory_, std::size_t buffer_size_, std::uint64_t remote_key_, unsigned iteration_count_, std::size_t msg_size_)
  {
    std::vector<client_state> v;
    for ( ; count_ != 0; --count_, ++remote_key_ )
    {
      v.emplace_back(factory_, buffer_size_, remote_key_, iteration_count_, msg_size_);
    }
    return v;
  }
}

pingpong_server_n::pingpong_server_n(
  unsigned client_count_
  , Component::IFabric_server_factory &factory_
  , std::uint64_t buffer_size_
  , std::uint64_t remote_key_base_
  , unsigned iteration_count_
  , std::uint64_t msg_size_
)
  : _cs(clients(client_count_, factory_, buffer_size_, remote_key_base_, iteration_count_, msg_size_))
  , _stat()
  , _th(
    &pingpong_server_n::listener
    , this
    , msg_size_
  )
{
}

pingpong_stat pingpong_server_n::time()
{
  if ( _th.joinable() )
  {
    _th.join();
  }
  return _stat;
}

pingpong_server_n::~pingpong_server_n()
try
{
  if ( _th.joinable() )
  {
    _th.join();
  }
}
catch ( std::exception &e )
{
  std::cerr << __func__ << " exception " << e.what() << eyecatcher << std::endl;
}

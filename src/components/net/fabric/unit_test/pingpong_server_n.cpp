#include "pingpong_server_n.h"

#include "eyecatcher.h"
#include <api/fabric_itf.h> /* IFabric_server_factory */
#include <common/errors.h> /* S_OK */
#include <gtest/gtest.h>

#include <exception>
#include <functional> /* ref */
#include <iostream> /* cerr */

void pingpong_server_n::listener(
  std::size_t /* msg_size_ */
)
try
{
  for ( auto &c : _cs )
  {
    c.sc.cnxn().post_recv(&*c.br[0].v.begin(), &*c.br[0].v.end(), &*c.br[0].d.begin(), &c.recv0_ctxt);
    c.sc.cnxn().post_recv(&*c.br[1].v.begin(), &*c.br[1].v.end(), &*c.br[1].d.begin(), &c.recv1_ctxt);
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
        c.sc.cnxn().poll_completions(cb_ctxt::cb);
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

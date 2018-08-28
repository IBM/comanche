#include "pingpong_server.h"

#include "eyecatcher.h"
#include "pingpong_server_client_state.h"
#include <exception>
#include <cstdint> /* uint64_t */
#include <chrono>
#include <functional> /* ref */
#include <iostream> /* cerr */

namespace Component
{
  class IFabric_server_factory;
}
void pingpong_server::listener(
  Component::IFabric_server_factory &factory_
  , std::size_t buffer_size_
  , std::uint64_t remote_key_
  , unsigned iteration_count_
  , std::size_t msg_size_
)
try
{
  client_state c{factory_, buffer_size_, remote_key_, iteration_count_, msg_size_};

  auto &st = c.st;
  st._comm.post_recv(&*st.br[0].v.begin(), &*st.br[0].v.end(), &*st.br[0].d.begin(), &st.recv0_ctxt);
  st._comm.post_recv(&*st.br[1].v.begin(), &*st.br[1].v.end(), &*st.br[1].d.begin(), &st.recv1_ctxt);

  std::uint64_t poll_count = 0U;
  auto polled_any = true;
  while ( polled_any )
  {
    polled_any = false;

    if ( st.iterations_left != 0 )
    {
      if ( _stat.start() == std::chrono::high_resolution_clock::time_point::min() )
      {
        _stat.do_start();
      }
      st._comm.poll_completions(cb_ctxt::cb);
      ++poll_count;
      polled_any = true;
    }
  }

  _stat.do_stop(poll_count);
}
catch ( std::exception &e )
{
  std::cerr << "pingpong_server::" << __func__ << ": " << e.what() << "\n";
  throw;
}

pingpong_server::pingpong_server(
  Component::IFabric_server_factory &factory_
  , std::uint64_t buffer_size_
  , std::uint64_t remote_key_base_
  , unsigned iteration_count_
  , std::uint64_t msg_size_
)
  : _stat{}
  , _th(
    &pingpong_server::listener
    , this
    , std::ref(factory_), buffer_size_, remote_key_base_, iteration_count_, msg_size_
  )
{
}

pingpong_stat pingpong_server::time()
{
  if ( _th.joinable() )
  {
    _th.join();
  }
  return _stat;
}

pingpong_server::~pingpong_server()
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

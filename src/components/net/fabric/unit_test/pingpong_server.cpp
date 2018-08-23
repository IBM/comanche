#include "pingpong_server.h"

#include "eyecatcher.h"
#include "registered_memory.h"
#include "wait_poll.h"
#include <api/fabric_itf.h> /* IFabric, IFabric_server_factory */
#include <common/errors.h> /* S_OK */
#include <gtest/gtest.h>
#include <sys/uio.h> /* iovec */

#include <exception>
#include <functional> /* ref */
#include <iostream> /* cerr */
#include <string>
#include <memory> /* make_shared, shared_ptr */
#include <thread>
#include <vector>

void pingpong_server::listener(
  std::size_t buffer_size_
  , std::uint64_t remote_key_
  , unsigned iteration_count_
  , std::size_t msg_size_
)
try
{
  EXPECT_LT(0U, _sc.cnxn().max_message_size());
  /* register an RDMA memory region */
  registered_memory rm{_sc.cnxn(), buffer_size_, remote_key_};
  /* wait for client indicate exit (by sending one byte to us) */
  std::vector<::iovec> v{{&rm[0], msg_size_}};
  std::vector<void *> d{{rm.desc()}};
  _start = std::chrono::high_resolution_clock::now();
  for ( auto i = 0U ; i != iteration_count_ ; ++i )
  {
    _sc.cnxn().post_recv(&*v.begin(), &*v.end(), &*d.begin(), this);
    ::wait_poll(
      _sc.cnxn()
      , [&v, msg_size_, this] (void *ctxt_, ::status_t stat_, std::uint64_t, std::size_t len_, void *) -> void
        {
          ASSERT_EQ(ctxt_, this);
          ASSERT_EQ(stat_, S_OK);
          EXPECT_EQ(len_, msg_size_);
        }
      , test_type::performance
    );
    _sc.cnxn().post_send(&*v.begin(), &*v.end(), &*d.begin(), this);
    ::wait_poll(
      _sc.cnxn()
      , [&v, this] (void *ctxt_, ::status_t stat_, std::uint64_t, std::size_t, void *) -> void
        {
          ASSERT_EQ(ctxt_, this);
          ASSERT_EQ(stat_, S_OK);
        }
      , test_type::performance
    );
  }
  _stop = std::chrono::high_resolution_clock::now();
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
  : _sc(factory_)
  , _start()
  , _stop()
  , _th(
    &pingpong_server::listener
    , this
    , buffer_size_, remote_key_base_, iteration_count_, msg_size_
  )
{
}

std::pair<std::chrono::high_resolution_clock::time_point, std::chrono::high_resolution_clock::time_point> pingpong_server::time()
{
  if ( _th.joinable() )
  {
    _th.join();
  }
  return { _start, _stop };
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

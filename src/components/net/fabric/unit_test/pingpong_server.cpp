#include "pingpong_server.h"

#include "eyecatcher.h"
#include "registered_memory.h"
#include "server_connection.h"
#include "server_connection_and_memory.h"
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
  Component::IFabric_server_factory &ep_
  , std::uint64_t remote_key_
  , unsigned iteration_count_
  , std::size_t buffer_size_
)
try
{
  server_connection sc(ep_);
  EXPECT_EQ(sc.cnxn().max_message_size(), this->max_message_size());
  /* register an RDMA memory region */
  registered_memory rm{sc.cnxn(), remote_key_};
  /* wait for client indicate exit (by sending one byte to us) */
  std::vector<::iovec> v{{&rm[0], buffer_size_}};
  std::vector<void *> d{{rm.desc()}};
  for ( auto i = 0U ; i != iteration_count_ ; ++i )
  {
    sc.cnxn().post_recv(&*v.begin(), &*v.end(), &*d.begin(), this);
    ::wait_poll(
      sc.cnxn()
      , [&v, buffer_size_, this] (void *ctxt_, ::status_t stat_, std::uint64_t, std::size_t len_, void *) -> void
        {
          ASSERT_EQ(ctxt_, this);
          ASSERT_EQ(stat_, S_OK);
          EXPECT_EQ(len_, buffer_size_);
        }
      , test_type::performance
    );
    sc.cnxn().post_send(&*v.begin(), &*v.end(), &*d.begin(), this);
    ::wait_poll(
      sc.cnxn()
      , [&v, buffer_size_, this] (void *ctxt_, ::status_t stat_, std::uint64_t, std::size_t, void *) -> void
        {
          ASSERT_EQ(ctxt_, this);
          ASSERT_EQ(stat_, S_OK);
#if 0
          EXPECT_EQ(len_, buffer_size_);
#endif
        }
      , test_type::performance
    );
  }
}
catch ( std::exception &e )
{
  std::cerr << "pingpong_server::" << __func__ << ": " << e.what() << "\n";
  throw;
}

pingpong_server::pingpong_server(
  Component::IFabric &fabric_
  , const std::string &fabric_spec_
  , std::uint16_t control_port_
  , std::uint64_t remote_key_base_
  , unsigned iteration_count_
  , std::uint64_t buffer_size_
)
  : _ep(fabric_.open_server_factory(fabric_spec_, control_port_))
  , _th(&pingpong_server::listener, this, std::ref(*_ep), remote_key_base_, iteration_count_, buffer_size_)
{
}

pingpong_server::~pingpong_server()
try
{
  _th.join();
}
catch ( std::exception &e )
{
  std::cerr << __func__ << " exception " << e.what() << eyecatcher << std::endl;
}

std::size_t pingpong_server::max_message_size() const
{
  return _ep->max_message_size();
}

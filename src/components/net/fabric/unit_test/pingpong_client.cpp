#include "pingpong_client.h"

#include "eyecatcher.h" /* pingpong_offset */
#include "patience.h" /* open_connection_patiently */
#include "registered_memory.h"
#include "wait_poll.h"

#include <api/fabric_itf.h> /* _Fabric_client */
#include <common/errors.h> /* S_OK */
#include <common/types.h> /* status_t */
#include <gtest/gtest.h>
#include <sys/uio.h> /* iovec */
#include <cstring> /* memcpy */
#include <exception>
#include <iostream> /* cerr */
#include <string>
#include <memory> /* make_shared */
#include <vector>

void pingpong_client::check_complete_static(void *t_, void *ctxt_, ::status_t stat_)
try
{
  /* The callback context must be the object which was polling. */
  ASSERT_EQ(t_, ctxt_);
  auto rmc = static_cast<pingpong_client *>(ctxt_);
  EXPECT_TRUE(rmc);
  EXPECT_EQ(stat_, ::S_OK);
}
catch ( std::exception &e )
{
  std::cerr << "pingpong_client::" << __func__ << e.what() << "\n";
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
  , _start{}
  , _stop{}
{
  registered_memory rm{*_cnxn, buffer_size_, remote_key_base_};
  std::vector<::iovec> v{{&rm[0], msg_size_}};
  std::vector<void *> d{{rm.desc()}};
  _start = std::chrono::high_resolution_clock::now();
  for ( auto i = 0U ; i != iteration_count_ ; ++i )
  {
    _cnxn->post_send(&*v.begin(), &*v.end(), &*d.begin(), this);
    wait_poll(
        *_cnxn
      , [&v, this] (void *ctxt_, ::status_t stat_, std::uint64_t, std::size_t, void *) -> void
        {
          EXPECT_EQ(ctxt_, this);
          EXPECT_EQ(stat_, ::S_OK);
        }
      , test_type::performance
    );
    _cnxn->post_recv(&*v.begin(), &*v.end(), &*d.begin(), this);
    wait_poll(
        *_cnxn
      , [&v, msg_size_, this] (void *ctxt_, ::status_t stat_, std::uint64_t, std::size_t len_, void *) -> void
        {
          EXPECT_EQ(ctxt_, this);
          EXPECT_EQ(stat_, ::S_OK);
          EXPECT_EQ(len_, msg_size_);
        }
      , test_type::performance
    );
  }
  _stop = std::chrono::high_resolution_clock::now();
}
catch ( std::exception &e )
{
  std::cerr << __func__ << ": " << e.what() << "\n";
  throw;
}

std::pair<std::chrono::high_resolution_clock::time_point, std::chrono::high_resolution_clock::time_point> pingpong_client::time()
{
  return { _start, _stop };
}

pingpong_client::~pingpong_client()
{
}

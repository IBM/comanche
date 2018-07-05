#include "remote_memory_client_grouped.h"

#include "eyecatcher.h"
#include "patience.h"
#include "remote_memory_subclient.h"
#include "wait_poll.h"
#include <common/errors.h> /* S_OK */
#include <boost/io/ios_state.hpp>
#include <gtest/gtest.h>
#include <sys/uio.h> /* iovec */
#include <vector>

void remote_memory_client_grouped::check_complete_static(void *rmc_, ::status_t stat_)
 {
   auto rmc = static_cast<remote_memory_subclient *>(rmc_);
   ASSERT_TRUE(rmc);
    rmc->check_complete(stat_);
}

void remote_memory_client_grouped::check_complete_static_2(void *t_, void *rmc_, ::status_t stat_)
{
  /* The callback context must be the object which was polling. */
  ASSERT_EQ(t_, rmc_);
  check_complete_static(rmc_, stat_);
}

void remote_memory_client_grouped::check_complete(::status_t stat_)
{
  ASSERT_EQ(stat_, S_OK);
}

remote_memory_client_grouped::remote_memory_client_grouped(Component::IFabric &fabric_, const std::string &fabric_spec_, const std::string ip_address_, std::uint16_t port_)
  : _cnxn(open_connection_grouped_patiently(fabric_, fabric_spec_, ip_address_, port_))
  , _rm_out{std::make_shared<registered_memory>(*_cnxn)}
  , _rm_in{std::make_shared<registered_memory>(*_cnxn)}
  , _vaddr{}
  , _key{}
  , _quit_flag('n')
{
  std::vector<iovec> v;
  iovec iv;
  iv.iov_base = &rm_out()[0];
  iv.iov_len = (sizeof _vaddr) + (sizeof _key);
  v.emplace_back(iv);

 remote_memory_subclient rms(*this);
 auto &cnxn = rms.cnxn();

 cnxn.post_recv(v, this);
 wait_poll(
    cnxn
    , [&v, this] (void *ctxt, ::status_t st) -> void
      {
        ASSERT_EQ(ctxt, this);
        ASSERT_EQ(st, S_OK);
        ASSERT_EQ(v[0].iov_len, (sizeof _vaddr) + sizeof( _key));
        std::memcpy(&_vaddr, &rm_out()[0], sizeof _vaddr);
        std::memcpy(&_key, &rm_out()[sizeof _vaddr], sizeof _key);
      }
  );
  boost::io::ios_flags_saver sv(std::cerr);
  std::cerr << "Remote memory client addr " << reinterpret_cast<void*>(_vaddr) << " key " << std::hex << _key << std::endl;
}

remote_memory_client_grouped::~remote_memory_client_grouped()
try
{
  remote_memory_subclient rms(*this);
  auto &cnxn = rms.cnxn();
  send_disconnect(cnxn, rm_out(), _quit_flag);
}
catch ( std::exception &e )
{
  std::cerr << __func__ << " exception " << e.what() << eyecatcher << std::endl;
}

void remote_memory_client_grouped::send_disconnect(Component::IFabric_communicator &cnxn_, registered_memory &rm_, char quit_flag_)
{
  send_msg(cnxn_, rm_, &quit_flag_, sizeof quit_flag_);
}

Component::IFabric_communicator *remote_memory_client_grouped::allocate_group() const
{
  return _cnxn->allocate_group();
}

#include "remote_memory_accessor.h"

#include "eyecatcher.h"
#include "registered_memory.h"
#include "wait_poll.h"
#include <api/fabric_itf.h> /* IFabric_communicator */
#include <common/errors.h> /* S_OK */
#include <gtest/gtest.h>
#include <sys/uio.h> /* iovec */
#include <cstdint> /* uint64_t */
#include <cstring> /* memcpy */
#include <exception>
#include <iostream> /* cerr */
#include <vector>

void remote_memory_accessor::send_memory_info(Component::IFabric_communicator &cnxn_, registered_memory &rm_)
{
  std::uint64_t vaddr = reinterpret_cast<std::uint64_t>(&rm_[0]);
  std::uint64_t key = rm_.key();
  char msg[(sizeof vaddr) + (sizeof key)];
  std::memcpy(msg, &vaddr, sizeof vaddr);
  std::memcpy(&msg[sizeof vaddr], &key, sizeof key);
  send_msg(cnxn_, rm_, msg, sizeof msg);
}

void remote_memory_accessor::send_msg(Component::IFabric_communicator &cnxn_, registered_memory &rm_, const void *msg_, std::size_t len_)
{
  std::vector<::iovec> v;
  std::memcpy(&rm_[0], msg_, len_);
  ::iovec iv;
  iv.iov_base = &rm_[0];
  iv.iov_len = len_;
  v.emplace_back(iv);
  try
  {
    cnxn_.post_send(v, this);
    ::wait_poll(
      cnxn_
      , [&v, this] (void *ctxt, ::status_t st) -> void
        {
          ASSERT_EQ(ctxt, this);
          ASSERT_EQ(st, S_OK);
        }
    );
  }
  catch ( const std::exception &e )
  {
    std::cerr << __func__ << " exception " << e.what() << eyecatcher << std::endl;
  }
}

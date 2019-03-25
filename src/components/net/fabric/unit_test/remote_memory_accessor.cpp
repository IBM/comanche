/*
   Copyright [2017-2019] [IBM Corporation]
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include "remote_memory_accessor.h"

#include "eyecatcher.h"
#include "registered_memory.h"
#include "wait_poll.h"
#include <api/fabric_itf.h> /* IFabric_communicator */
#include <common/errors.h> /* S_OK */
#include <boost/io/ios_state.hpp>
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
  {
    boost::io::ios_flags_saver sv(std::cerr);
    std::cerr << "Server: memory addr " << reinterpret_cast<void*>(vaddr) << std::hex << " key " << key << "\n";
  }
  char msg[(sizeof vaddr) + (sizeof key)];
  std::memcpy(msg, &vaddr, sizeof vaddr);
  std::memcpy(&msg[sizeof vaddr], &key, sizeof key);
  send_msg(cnxn_, rm_, msg, sizeof msg);
}

void remote_memory_accessor::send_msg(Component::IFabric_communicator &cnxn_, registered_memory &rm_, const void *msg_, std::size_t len_)
{
  std::memcpy(&rm_[0], msg_, len_);
  std::vector<::iovec> v{{&rm_[0],len_}};
  std::vector<void *> d{rm_.desc()};
  try
  {
    cnxn_.post_send(&*v.begin(), &*v.end(), &*d.begin(), this);
    ::wait_poll(
      cnxn_
      , [this] (void *ctxt_, ::status_t stat_, std::uint64_t, std::size_t, void *) -> void
        {
          EXPECT_EQ(ctxt_, this);
          EXPECT_EQ(stat_, S_OK);
        }
    );
  }
  catch ( const std::exception &e )
  {
    std::cerr << __func__ << " exception " << e.what() << eyecatcher << std::endl;
  }
}

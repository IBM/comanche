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
#ifndef __SERVICE_H__
#define __SERVICE_H__

#include <core/ipc.h>
#include "service.pb.h"

class Service : public Core::IPC_server
{
private:
  static constexpr bool option_DEBUG = true;
public:
  Service(const std::string& instance_id) :
    Core::IPC_server(std::string("ipc:///tmp//ipc_memory_") + instance_id)
  {
    TRACE();
    GOOGLE_PROTOBUF_VERIFY_VERSION;
  }
  ~Service() {
    TRACE();
  }

private:
  int process_message(void* msg, size_t msg_len, void* reply, size_t reply_len);

};

#endif

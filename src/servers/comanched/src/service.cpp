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
#include <common/cycles.h>

#include "service.h"

int Service::process_message(void* buffer, size_t buffer_len, void* reply, size_t reply_len) {
  PLOG("process_message:");

  using namespace ipc_protocol;
  Ack reply_msg;
  int retval = 0; // 0: loop continue -1: loop exit
  
  /**
   * Google proto buf binary packing for embedded types is same as string
   * which means we skip the first two bytes (3 from the start?)
   *
   */
  uint32_t type = *((uint32_t*)&((char*)buffer)[3]);
  
  switch (type) {
  case MessageType::__CreateSessionRequest: {
    CreateSessionRequest msg;
    msg.ParseFromArray(buffer, buffer_len);
      
    /* TODO: verify auth */
    PLOG("CreateSessionRequest");
    PLOG("auth_id = (%s)", msg.hdr().auth_id().c_str());
    PLOG("auth_token = (%s)", msg.hdr().auth_token().c_str());

    CreateSessionReply reply_msg;
    reply_msg.mutable_hdr()->set_type(MessageType::__CreateSessionReply);
    reply_msg.mutable_hdr()->set_auth_id("heap_nvme_proc");
    reply_msg.mutable_hdr()->set_auth_token("machineAuthTok");
    reply_msg.mutable_hdr()->set_debug_code((uint32_t)(rdtsc() & 0xFFFFFFFFUL));
    break;
  }
    //-----------------------------------------------------------
  case MessageType::__ShutdownSessionRequest: {
    PLOG("Received ShutdownSession slow-path IPC message");
    ShutdownSession msg;
    msg.ParseFromArray(buffer, buffer_len);

    /* TODO verify auth */
    // signal shutdown
    retval = -1;
    break;
  }

    // default: {
    //   PERR("unknown session type! (%d)", .session_type());
    //   assert(0);
    // }
    
  }

  /* ack the message */
  reply_msg.mutable_hdr()->set_type(ipc_protocol::MessageType::__Ack);
  reply_msg.mutable_hdr()->set_auth_id("heap_nvme_proc");
  reply_msg.mutable_hdr()->set_auth_token("machineAuthTok");
  reply_msg.set_code(0);
  reply_msg.SerializeToArray(reply, reply_len);

  return retval;

  

}

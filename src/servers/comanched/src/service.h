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

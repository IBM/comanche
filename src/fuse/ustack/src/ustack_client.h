#ifndef __USTACK_CLIENT_H__
#define __USTACK_CLIENT_H__

#include <assert.h>
#include <core/ipc.h>
#include <core/uipc.h>

#include "protocol_generated.h"
#include "protocol_channel.h"

class Ustack_client: public Core::IPC_client
{
public:
  Ustack_client(const std::string endpoint) : Core::IPC_client(endpoint) {
  }

  ~Ustack_client() {
    /* send shutdown message */
    send_shutdown();
    
    for(auto& s: _shmem)
      delete s;

    if(_channel)
      delete _channel;
  }

  void send_shutdown() {
    using namespace Protocol;
    using namespace flatbuffers;
    flatbuffers::FlatBufferBuilder fbb(256);

    auto msg = CreateMessage(fbb,
                             MessageType_Shutdown,
                             getpid());
    
    FinishMessageBuffer(fbb, msg);
    send_no_wait((const char *) fbb.GetBufferPointer(), fbb.GetSize());
  }

  void get_io_memory(int n_pages) {
    using namespace Protocol;
    using namespace flatbuffers;
    flatbuffers::FlatBufferBuilder fbb(256);

    assert(n_pages > 0);
    auto msg = CreateMessage(fbb,
                             MessageType_IO_buffer_request,
                             getpid(),
                             Element_ElementIOBufferRequest,
                             CreateElementIOBufferRequest(fbb, n_pages).Union());
    
    FinishMessageBuffer(fbb, msg);

    assert(fbb.GetSize() > 0);
    size_t reply_len = 0;
    void * reply = send_and_wait((const char *) fbb.GetBufferPointer(),
                                 fbb.GetSize(),
                                 &reply_len);

    const Message * reply_msg = Protocol::GetMessage(reply);
    if(reply_msg->type() != MessageType_IO_buffer_reply)
      throw General_exception("bad response to Memory_request");
    
    if(reply_msg->element_type() == Element_ElementIOBufferReply) {
      addr_t phys_addr = reply_msg->element_as_ElementIOBufferReply()->phys_addr(); 
      PLOG("response: phys=%lx", phys_addr);
      PLOG("IO buffer acquired OK");
    }
    else throw General_exception("unexpected reply message");    
  }

  void get_uipc_channel() {
    using namespace Protocol;
    using namespace flatbuffers;
    flatbuffers::FlatBufferBuilder fbb(256);

    auto msg = CreateMessage(fbb,
                             MessageType_Channel_request,
                             getpid());
    
    FinishMessageBuffer(fbb, msg);

    assert(fbb.GetSize() > 0);
    size_t reply_len = 0;
    void * reply = send_and_wait((const char *) fbb.GetBufferPointer(),
                                 fbb.GetSize(),
                                 &reply_len);

    const Message * reply_msg = Protocol::GetMessage(reply);
    if(reply_msg->type() != MessageType_Channel_reply)
      throw General_exception("bad response to Memory_request");
    
    if(reply_msg->element_type() == Element_ElementChannelReply) {
      std::string channel_id = reply_msg->element_as_ElementChannelReply()->uipc_id()->str();
      PLOG("response: channel(%s)", channel_id.c_str());
      _channel = new Core::UIPC::Channel(channel_id);
      PLOG("channel acquired (%s)", channel_id.c_str());
    }
    else throw General_exception("unexpected reply message");    

  }
  
  void get_shared_memory(size_t n_bytes) {
    using namespace Protocol;
    using namespace flatbuffers;
    flatbuffers::FlatBufferBuilder fbb(256);

    auto msg = CreateMessage(fbb,
                             MessageType_Memory_request,
                             getpid(),
                             Element_ElementMemoryRequest,
                             CreateElementMemoryRequest(fbb, n_bytes).Union());
    
    FinishMessageBuffer(fbb, msg);
    
    assert(fbb.GetSize() > 0);
    size_t reply_len = 0;
    void * reply = send_and_wait((const char *) fbb.GetBufferPointer(),
                                 fbb.GetSize(),
                                 &reply_len);

    const Message * reply_msg = Protocol::GetMessage(reply);
    if(reply_msg->type() != MessageType_Memory_reply)
      throw General_exception("bad response to Memory_request");
    
    if(reply_msg->element_type() == Element_ElementMemoryReply) {
      std::string shmem_id = reply_msg->element_as_ElementMemoryReply()->shmem_id()->str();
      PLOG("response: %s", shmem_id.c_str());
      _shmem.push_back(new Core::UIPC::Shared_memory(shmem_id));
      PLOG("shared memory acquired (%s)", shmem_id.c_str());
    }
    else throw General_exception("unexpected reply message");    
  }

  void send_command()
  {
    assert(_channel);
    struct IO_command * cmd = static_cast<struct IO_command *>(_channel->alloc_msg());
    cmd->type = 99;
    strcpy(cmd->data, "hello");
    _channel->send(cmd);

    void * reply = nullptr;
    while(_channel->recv(reply));
    PLOG("waiting for IO channel reply...");
    _channel->free_msg(reply);
    PLOG("send command and got reply.");
  }

private:
  std::vector<Core::UIPC::Shared_memory *> _shmem;
  Core::UIPC::Channel *                    _channel = nullptr;
};


#endif // __USTACK_CLIENT_H__

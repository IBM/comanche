#ifndef __USTACK_H__
#define __USTACK_H__

#include <map>
#include <thread>
#include <core/ipc.h>
#include <core/uipc.h>
#include <tbb/concurrent_vector.h>
#include <api/block_itf.h>
#include <api/region_itf.h>
#include <api/blob_itf.h>

#include "protocol_channel.h"

class Ustack : public Core::IPC_server {
public:
  Ustack(const std::string endpoint);
  ~Ustack();
public:
  Component::IBlock_device * block = nullptr;
  Component::IBlob * store = nullptr;

protected:
  virtual int process_message(void* msg,
                              size_t msg_len,
                              void* reply,
                              size_t reply_len);

  virtual void post_reply(void * reply_msg);

private:

  void uipc_channel_thread_entry(Core::UIPC::Channel * channel);
  void release_resources(pid_t client_id);
  
  struct Shared_memory_instance
  {
    Shared_memory_instance(std::string id, size_t size, unsigned long client_id) :
      _id(id), _size(size), _client_id(client_id), _shmem(nullptr) {
    }
      
    std::string                 _id;
    size_t                      _size;
    pid_t                       _client_id;
    Core::UIPC::Shared_memory * _shmem;
  };

  struct Channel_instance
  {
    Channel_instance(std::string id, pid_t client_id) :
      _id(id), _client_id(client_id), _channel(nullptr) {
    }
      
    std::string           _id;
    pid_t                 _client_id;
    Core::UIPC::Channel * _channel;
  };

  static constexpr unsigned MAX_MESSAGE_SIZE = sizeof(IO_command);
  static constexpr unsigned MESSAGE_QUEUE_SIZE = 16;

  //  typedef unsigned long pid_t;
  bool _shutdown = false;
  std::thread *                                          _ipc_thread;
  std::vector<Shared_memory_instance*>                   _pending_shmem;
  std::vector<Channel_instance*>                         _pending_channels;

  std::map<pid_t, std::vector<std::thread *>>            _threads;
  std::map<pid_t, std::vector<Shared_memory_instance *>> _shmem_map;
  std::map<pid_t, Channel_instance *>                    _channel_map;
};


#endif 

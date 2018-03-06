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
  
  struct Shared_memory_instance
  {
    Shared_memory_instance(std::string id, size_t size, unsigned long client_id) :
      _id(id), _size(size), _client_id(client_id), _shmem(nullptr) {
    }
      
    std::string _id;
    size_t _size;
    unsigned long _client_id;
    Core::UIPC::Shared_memory * _shmem;
  };


  std::thread *                                                  _ipc_thread;
  std::vector<Shared_memory_instance*>                           _pending;
  std::map<unsigned long, std::vector<Shared_memory_instance *>> _shmem_map;
};


#endif 

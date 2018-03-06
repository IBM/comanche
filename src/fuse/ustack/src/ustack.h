#ifndef __USTACK_H__
#define __USTACK_H__

#include <thread>
#include <core/ipc.h>

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

private:
  std::thread * _ipc_thread;
};


#endif 

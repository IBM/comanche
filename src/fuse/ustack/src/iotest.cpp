#include <iostream>
#include <fstream>
#include <stdio.h>
#include <assert.h>
#include <core/ipc.h>
#include <core/uipc.h>

#include "protocol_generated.h"

class Ustack_client: public Core::IPC_client
{
public:
  Ustack_client(const std::string endpoint) : Core::IPC_client(endpoint) {
  }

  ~Ustack_client() {
    /* TODO: send disconnect message */
    for(auto& s: _shmem)
      delete s;
  }

  
  const std::string get_shared_memory_id() {
    using namespace Protocol;
    using namespace flatbuffers;
    flatbuffers::FlatBufferBuilder fbb(1024);

    auto msg = CreateMessage(fbb,
                             getpid(),
                             Element_ElementMemoryRequest,
                             CreateElementMemoryRequest(fbb, 1222).Union());
    
    FinishMessageBuffer(fbb, msg);
    
    assert(fbb.GetSize() > 0);
    size_t reply_len = 0;
    void * reply = send_and_wait((const char *) fbb.GetBufferPointer(),
                                 fbb.GetSize(),
                                 &reply_len);

    const Message * reply_msg = Protocol::GetMessage(reply);
    if(reply_msg->element_type() == Element_ElementMemoryReply) {
      std::string shmem_id = reply_msg->element_as_ElementMemoryReply()->shmem_id()->c_str();
      PLOG("response: %s", shmem_id.c_str());
      _shmem.push_back(new Core::UIPC::Shared_memory(shmem_id));
      PLOG("shared memory acquired (%s)", shmem_id.c_str());
    }
    else throw General_exception("unexpected reply message");    
  }

private:
  std::vector<Core::UIPC::Shared_memory *> _shmem;

};

int main()
{
  Ustack_client ustack("ipc:///tmp//ustack.ipc");

  ustack.get_shared_memory_id();
  
  FILE * fp = fopen("./fs/fio.blob","w+");
  if(fp==NULL) {
    perror("error:");
  }

  //  char buf[256];
  //  size_t rc = fread(buf, 4096, 1, fp);
  
  #if 0
  
  for(unsigned i=0;i<100;i++) {
    size_t rc = fread(buf, 256, 1, fp);
    assert(rc == 256);
  }
  #endif
  
  fclose(fp);
  return 0;
}

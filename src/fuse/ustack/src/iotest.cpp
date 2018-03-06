#include <iostream>
#include <fstream>
#include <stdio.h>
#include <assert.h>
#include <core/ipc.h>

#include "protocol_generated.h"

class Ustack_client: public Core::IPC_client
{
public:
  Ustack_client(const std::string endpoint) : Core::IPC_client(endpoint) {
  }
  
  const std::string get_shared_memory_id() {
    using namespace Protocol;
    flatbuffers::FlatBufferBuilder fbb(1024);
    MessageBuilder builder(fbb);    
    builder.add_sender_id(999);

    auto element = CreateElementMemoryRequest(fbb, MessageType_Memory_request);
    builder.add_element(element.Union());
    auto msg = builder.Finish();

    size_t reply_len = 0;
    void * reply = send_and_wait((const char *) fbb.GetBufferPointer(),
                                 fbb.GetSize(),
                                 &reply_len);
  }
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

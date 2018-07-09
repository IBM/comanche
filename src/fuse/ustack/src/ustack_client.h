#ifndef __USTACK_CLIENT_H__
#define __USTACK_CLIENT_H__

#include <assert.h>
#include <core/ipc.h>
#include <core/uipc.h>
#include <core/avl_malloc.h>
#include <core/xms.h>

#include "protocol_generated.h"
#include "protocol_channel.h"

/*
 * IO Memory mapper for client.
 *
 * Client maps the iomem to its virtual space after get reply of IO_buffer_request
 */
class IO_memory_allocator : private Core::Region_allocator
{
public:
  IO_memory_allocator(addr_t phys_base, size_t n_bytes) :
    _phys_base(phys_base),    
    Core::Region_allocator(phys_base, n_bytes) {

    void * ptr = xms_mmap((void*) 0x1100000000, phys_base, n_bytes);
    assert(ptr);
    _virt_base = reinterpret_cast<addr_t>(ptr);
    _offset = _virt_base - _phys_base;

    PMAJOR("IO_memory_allocator: %ld bytes (%p)", n_bytes, ptr);
  }

  ~IO_memory_allocator() {
    munmap(reinterpret_cast<void*>(_virt_base), _size);
  }

  void * malloc(size_t n_bytes) {
    return get_virt(alloc(n_bytes, 8 /* alignment */));
  }

  void free(void * ptr) {
    return Core::Region_allocator::free(get_phys(ptr));
  }

  void * get_virt(addr_t paddr) {
    return reinterpret_cast<void*>(paddr + _offset);
  }

  addr_t get_phys(void * vaddr) {
    return reinterpret_cast<addr_t>(vaddr) - _offset;
  }

private:
  addr_t _virt_base;
  addr_t _phys_base;
  ssize_t _offset;
  size_t _size;
  
};

class Ustack_client: public Core::IPC_client
{
public:
  /*
   * Construct
   * 
   * io memory will be allocated in server and mapped locally
   */
  Ustack_client(const std::string endpoint, size_t n_iopages = 64) :
    Core::IPC_client(endpoint),
    _iomem_allocator(get_io_memory(n_iopages),n_iopages * PAGE_SIZE)
  {
    PLOG("Ustack: allocating %ld pages of IO memory", n_iopages);
    
  }

  ~Ustack_client() {
    /* send shutdown message */
    send_shutdown();
    
    for(auto& s: _shmem)
      delete s;

    if(_channel)
      delete _channel;

    PLOG("UStack deallocated");
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

  addr_t get_io_memory(int n_pages) {
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

    addr_t phys_addr;
    if(reply_msg->element_type() == Element_ElementIOBufferReply) {
      phys_addr = reply_msg->element_as_ElementIOBufferReply()->phys_addr(); 
      PLOG("response: phys=%lx", phys_addr);
      PLOG("IO buffer acquired OK");
    }
    else throw General_exception("unexpected reply message");
    return phys_addr;
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

  /*
   * file write and read
   * TODO: intercept posix ops
   *
   * the message will be like(fd, phys(buf), count)
   */
  size_t write(int fd, const void *buf, size_t count){
    PLOG("[stack-write]: try to write from %p to fd %d, size %lu", buf, fd, count);
    assert(_channel);
    struct IO_command * cmd = static_cast<struct IO_command *>(_channel->alloc_msg());

    cmd->type = IO_TYPE_WRITE;
    cmd->offset = 0;

    //strcpy(cmd->data, "hello");
    _channel->send(cmd);

    void * reply = nullptr;
    while(_channel->recv(reply));
    PLOG("waiting for IO channel reply...");
    PLOG("get IO channel reply with type %d", static_cast<struct IO_command *>(reply)->type);
    _channel->free_msg(reply);
    PLOG("send command and got reply.");

    return 0;
  }

  size_t read(int fd, void *buf, size_t count){
    return 0;
  }

  /* 
   * passthrough memory management
   **/
  void * malloc(size_t n_bytes) {
    return _iomem_allocator.malloc(n_bytes);
  }

  void free(void * ptr) {
    return _iomem_allocator.free(ptr);
  }

  void * get_virt(addr_t paddr) {
    return _iomem_allocator.get_virt(paddr);
  }

  addr_t get_phys(void * vaddr) {
    return _iomem_allocator.get_phys(vaddr);
  }


private:
  IO_memory_allocator                      _iomem_allocator;
  std::vector<Core::UIPC::Shared_memory *> _shmem;
  Core::UIPC::Channel *                    _channel = nullptr;
};


#endif // __USTACK_CLIENT_H__

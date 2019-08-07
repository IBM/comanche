#ifndef __USTACK_CLIENT_H__
#define __USTACK_CLIENT_H__

#include <assert.h>
#include <core/avl_malloc.h>
#include <core/ipc.h>
#include <core/uipc.h>
#include <core/xms.h>
#include <dlfcn.h>
#include <unordered_map>

#include "protocol_channel.h"
#include "protocol_generated.h"
#include "ustack_client_ioctl.h"

/**
 * IO Memory mapper for client.
 *
 * Client maps the iomem to its virtual space after get reply of
 * IO_buffer_request.
 */
class IO_memory_allocator : private Core::Region_allocator {
  // std::string MOUNT_ROOT="/home/fengggli/comanche/build/mymount";

 public:
  IO_memory_allocator(addr_t phys_base, size_t n_bytes)
      : _phys_base(phys_base), Core::Region_allocator(phys_base, n_bytes)
  {
    void *ptr = xms_mmap((void *) 0x1100000000, phys_base, n_bytes);
    assert(ptr);
    _virt_base        = reinterpret_cast<addr_t>(ptr);
    _offset_virt_phys = _virt_base - _phys_base;
    _size             = n_bytes;

    PMAJOR("IO_memory_allocator: %ld bytes (%p)", n_bytes, ptr);
  }

  ~IO_memory_allocator()
  {
    munmap(reinterpret_cast<void *>(_virt_base), _size);
  }

  void *malloc(size_t n_bytes)
  {
    return get_virt(alloc(n_bytes, 8 /* alignment */));
  }

  status_t free(void *ptr)
  {
    PLOG("try to free");
    addr_t phys_addr = get_phys(ptr);
    if (Core::Region_allocator::contains(phys_addr)) {
      Core::Region_allocator::free(phys_addr);
      return S_OK;
    }
    else
      return E_FAIL;
  }

  void *get_virt(addr_t paddr)
  {
    return reinterpret_cast<void *>(paddr + _offset_virt_phys);
  }

  addr_t get_phys(void *vaddr)
  {
    return reinterpret_cast<addr_t>(vaddr) - _offset_virt_phys;
  }

  size_t get_offset(const void *vaddr)
  {
    return reinterpret_cast<addr_t>(vaddr) - _virt_base;
  }

 private:
  addr_t  _virt_base;
  addr_t  _phys_base;
  ssize_t _offset_virt_phys;
  size_t  _size;
};

class Ustack_client : public Core::IPC_client {
 public:
  /*
   * Construct
   *
   * io memory will be allocated in server and mapped locally
   */
  Ustack_client(const std::string endpoint, size_t n_iopages = 64)
      : Core::IPC_client(endpoint),
        _iomem_allocator(get_io_memory(n_iopages), n_iopages * PAGE_SIZE)
  {
    PLOG("Ustack: allocating %ld pages of IO memory", n_iopages);
  }

  ~Ustack_client()
  {
    /* send shutdown message */
    send_shutdown();

    for (auto &s : _shmem) delete s;

    if (_channel) delete _channel;

    PLOG("UStack deallocated");
  }

  void send_shutdown()
  {
    using namespace Protocol;
    using namespace flatbuffers;
    flatbuffers::FlatBufferBuilder fbb(256);

    auto msg = CreateMessage(fbb, MessageType_Shutdown, getpid());

    FinishMessageBuffer(fbb, msg);
    send_no_wait((const char *) fbb.GetBufferPointer(), fbb.GetSize());
  }

  addr_t get_io_memory(int n_pages)
  {
    using namespace Protocol;
    using namespace flatbuffers;
    flatbuffers::FlatBufferBuilder fbb(256);

    assert(n_pages > 0);
    auto msg =
        CreateMessage(fbb, MessageType_IO_buffer_request, getpid(),
                      Element_ElementIOBufferRequest,
                      CreateElementIOBufferRequest(fbb, n_pages).Union());

    FinishMessageBuffer(fbb, msg);

    assert(fbb.GetSize() > 0);
    size_t reply_len = 0;
    void * reply     = send_and_wait((const char *) fbb.GetBufferPointer(),
                                fbb.GetSize(), &reply_len);

    const Message *reply_msg = Protocol::GetMessage(reply);
    if (reply_msg->type() != MessageType_IO_buffer_reply)
      throw General_exception("bad response to Memory_request");

    addr_t phys_addr;
    if (reply_msg->element_type() == Element_ElementIOBufferReply) {
      phys_addr = reply_msg->element_as_ElementIOBufferReply()->phys_addr();
      PLOG("response: phys=%lx", phys_addr);
      PLOG("IO buffer acquired OK");
    }
    else
      throw General_exception("unexpected reply message");
    return phys_addr;
  }

  Core::UIPC::Channel *get_uipc_channel()
  {
    using namespace Protocol;
    using namespace flatbuffers;
    flatbuffers::FlatBufferBuilder fbb(256);

    auto msg = CreateMessage(fbb, MessageType_Channel_request, getpid());

    FinishMessageBuffer(fbb, msg);

    assert(fbb.GetSize() > 0);
    size_t reply_len = 0;
    PDBG("[client]: sent  channel request");
    void *reply = send_and_wait((const char *) fbb.GetBufferPointer(),
                                fbb.GetSize(), &reply_len);

    const Message *reply_msg = Protocol::GetMessage(reply);
    if (reply_msg->type() != MessageType_Channel_reply)
      throw General_exception("bad response to Memory_request");

    PDBG("[client]: got first channel reply");

    if (reply_msg->element_type() == Element_ElementChannelReply) {
      std::string channel_id =
          reply_msg->element_as_ElementChannelReply()->uipc_id()->str();
      PLOG("response: channel(%s)", channel_id.c_str());
      // This will wait till server side channel is up and then connects
      _channel = new Core::UIPC::Channel(channel_id);
      PLOG("channel acquired (%s)", channel_id.c_str());
    }
    else
      throw General_exception("unexpected reply message");
    PMAJOR("[client]: channel connected");
    return _channel;
  }

  void get_shared_memory(size_t n_bytes)
  {
    using namespace Protocol;
    using namespace flatbuffers;
    flatbuffers::FlatBufferBuilder fbb(256);

    auto msg = CreateMessage(fbb, MessageType_Memory_request, getpid(),
                             Element_ElementMemoryRequest,
                             CreateElementMemoryRequest(fbb, n_bytes).Union());

    FinishMessageBuffer(fbb, msg);

    assert(fbb.GetSize() > 0);
    size_t reply_len = 0;
    void * reply     = send_and_wait((const char *) fbb.GetBufferPointer(),
                                fbb.GetSize(), &reply_len);

    const Message *reply_msg = Protocol::GetMessage(reply);
    if (reply_msg->type() != MessageType_Memory_reply)
      throw General_exception("bad response to Memory_request");

    if (reply_msg->element_type() == Element_ElementMemoryReply) {
      std::string shmem_id =
          reply_msg->element_as_ElementMemoryReply()->shmem_id()->str();
      PLOG("response: shmem(%s)", shmem_id.c_str());
      _shmem.push_back(new Core::UIPC::Shared_memory(shmem_id));
      PLOG("shared memory acquired (%s)", shmem_id.c_str());
    }
    else
      throw General_exception("unexpected reply message");
    PMAJOR("[client]: got shared memory");
  }

  void send_command()
  {
    assert(_channel);
    struct IO_command *cmd =
        static_cast<struct IO_command *>(_channel->alloc_msg());
    cmd->type = 99;
    strcpy(cmd->data, "hello");
    _channel->send(cmd);

    void *reply = nullptr;
    PLOG("waiting for IO channel reply...");
    while (_channel->recv(reply))
      ;
    _channel->free_msg(reply);
    PLOG("send command and got reply.");
  }

  /********************************
   * Memory management
   ********************************/
  void *malloc(size_t n_bytes)
  {
    void *virt_addr;
    try {
      virt_addr = _iomem_allocator.malloc(n_bytes);
    }
    catch (...) {
      PERR("ustack client malloc error");
      return nullptr;
    }

    return virt_addr;
  }

  void free(void *ptr) { _iomem_allocator.free(ptr); }

  /**
   * File write and read
   * TODO: intercept posix ops
   *
   * the message will be like(fd, phys(buf), count)
   */
  ssize_t write(int fuse_fh, const void *buf, size_t count, off_t file_off)
  {
    int ret = -1;
    /* ustack tracked file */
    assert(_channel);
    struct IO_command *cmd =
        static_cast<struct IO_command *>(_channel->alloc_msg());

    // TODO: local cache of the fd->fuse-fd?
    cmd->fuse_fh  = fuse_fh;
    cmd->type     = IO_TYPE_WRITE;
    cmd->offset   = _iomem_allocator.get_offset(buf);
    cmd->sz_bytes = count;
    cmd->file_off = file_off;

    if(count > GB(4)) PWRN("32bit field for sz_bytes is not enough");

    // strcpy(cmd->data, "hello");
    _channel->send(cmd);

    void *reply = nullptr;
    PLOG("waiting for IO channel reply...");
    while (_channel->recv(reply))
      ;
    // PLOG("get IO channel reply with type %d", static_cast<struct IO_command
    // *>(reply)->type);
    if (IO_WRITE_OK != static_cast<struct IO_command *>(reply)->type) {
      PERR("[%s]: ustack write failed", __func__);
      goto cleanup;
    }
    ret = count;
  cleanup:
    _channel->free_msg(reply);
    PLOG("send write command and got reply.");
    return ret;
  }

  ssize_t read(int fuse_fh, void *buf, size_t count, off_t file_off)
  {
    int ret = -1;
    /* ustack tracked file */
    assert(_channel);
    struct IO_command *cmd =
        static_cast<struct IO_command *>(_channel->alloc_msg());

    // TODO: local cache of the fd->fuse-fd?
    cmd->fuse_fh  = fuse_fh;
    cmd->type     = IO_TYPE_READ;
    cmd->offset   = _iomem_allocator.get_offset(buf);
    cmd->sz_bytes = count;
    cmd->file_off = file_off;

    // strcpy(cmd->data, "hello");
    _channel->send(cmd);

    void *reply = nullptr;
    while (_channel->recv(reply))
      ;
    PLOG("waiting for IO channel reply...");
    // PLOG("get IO channel reply with type %d", static_cast<struct IO_command
    // *>(reply)->type);
    if (IO_READ_OK != static_cast<struct IO_command *>(reply)->type) {
      PERR("[%s]: ustack write failed", __func__);
      goto cleanup;
    }
    ret = count;
  cleanup:
    _channel->free_msg(reply);
    PLOG("send read command and got reply.");
    return ret;
  }
  /*
  void * get_virt(addr_t paddr) {
    return _iomem_allocator.get_virt(paddr);
  }

  addr_t get_phys(void * vaddr) {
    return _iomem_allocator.get_phys(vaddr);
  }
  */

 private:
  IO_memory_allocator _iomem_allocator;  // IO memory allocated from server and
                                         // mmaped to this client
  std::vector<Core::UIPC::Shared_memory *> _shmem;
  Core::UIPC::Channel *                    _channel = nullptr;
};  // end of UStack_Client

#endif  // __USTACK_CLIENT_H__

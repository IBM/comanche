#ifndef __USTACK_H__
#define __USTACK_H__

#include <map>
#include <thread>
#include <core/ipc.h>
#include <core/uipc.h>
#include <core/physical_memory.h>
#include <tbb/concurrent_vector.h>
#include <api/block_itf.h>
#include <api/region_itf.h>
#include <api/blob_itf.h>

#include "protocol_channel.h"
#include "kv_ustack_info.h"

class Ustack : public Core::IPC_server {
public:
  Ustack(const std::string endpoint, KV_ustack_info * kv_ustack_info = nullptr);
  ~Ustack();
public:

protected:
  virtual int process_message(void* msg,
                              size_t msg_len,
                              void* reply,
                              size_t reply_len);

  virtual void post_reply(void * reply_msg);

private:

  /* TODO: client id should be accessible from the channel*/
  void uipc_channel_thread_entry(Core::UIPC::Channel * channel, pid_t client_id);
  void release_resources(pid_t client_id);

  /* actually do the io */
  // will be called through slow path
  // status_t do_kv_open();
  // status_t do_kv_close();

  /*
   * Do the write through the kvfs daemon
   *
   * @param client_id  client who issue this io, we need this to find the right iomem
   * @param fuse_fh fuse daemon file handler
   * @param offset offset of iobuffer inside the iomem
   * @param io_sz  I/O size
   */
  status_t do_kv_write(pid_t client_id, uint64_t fuse_fh, size_t offset, size_t io_sz ){
    PDBG("[%s]: fuse_fh=%lu, offset=%lu, io_sz=%lu", __func__, fuse_fh, offset, io_sz);

    //get the virtual address and issue the io
    void *buf; //the mapped io mem

    // TODO: need to control offset 
    auto iomem_list = _iomem_map[client_id] ;
    if(iomem_list.empty()){
      PERR("iomem for pid %d is empty", client_id);
      return -1;
    }
    auto iomem = iomem_list[0];

    buf = iomem->offset_to_virt(offset);
    //TODO: check file 
    if(buf == 0){
      PERR("mapped virtual address failed");
      return -1;
    }
    PDBG("mapped to virtual address %p", buf);

    if(io_sz >0){
      _kv_ustack_info->write(fuse_fh, buf, io_sz);

      _kv_ustack_info->set_item_size(fuse_fh, io_sz);
    }
    else PWRN("io size = 0");
    
    return S_OK;
  }

  status_t do_kv_read(pid_t client_id, uint64_t fuse_fh, size_t offset, size_t io_sz){
    PDBG("[%s]: fuse_fh=%lu, offset=%lu, io_sz=%lu", __func__, fuse_fh, offset, io_sz);
    void *buf; //the mapped io mem

    auto iomem_list = _iomem_map[client_id] ;
    if(iomem_list.empty()){
      PERR("iomem for pid %d is empty", client_id);
      return -1;
    }
    auto iomem = iomem_list[0];

    buf = iomem->offset_to_virt(offset);
    //TODO: check file 
    if(buf == 0){
      PERR("mapped virtual address failed");
      return -1;
    }
    PDBG("mapped to virtual address %p", buf);

    size_t file_size = _kv_ustack_info->get_item_size(fuse_fh);
    if(io_sz > file_size){
      PERR("write size larger than file size");
      return -1;
    }

    if(io_sz >0){
      _kv_ustack_info->read(fuse_fh, buf, io_sz);
    }
      else PWRN("io size = 0");
    return S_OK;
  }
  
  struct Shared_memory_instance
  {
    Shared_memory_instance(std::string id, size_t size, unsigned long client_id) :
      _id(id), _size(size), _client_id(client_id), _shmem(nullptr) {
    }

    ~Shared_memory_instance() {
      if(_shmem) delete _shmem;
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
    ~Channel_instance() {
      if(_channel) delete _channel;
    }
      
    std::string           _id;
    pid_t                 _client_id;
    Core::UIPC::Channel * _channel;
  };

  struct IO_memory_instance
  {
    IO_memory_instance(size_t n_pages) : _n_pages(n_pages) {
      _iob = _allocator.allocate_io_buffer(n_pages * PAGE_SIZE, PAGE_SIZE, Component::NUMA_NODE_ANY);
      _phys_addr = _allocator.phys_addr(_iob);
    }
    ~IO_memory_instance() {
      _allocator.free_io_buffer(_iob);
    }

    /* offset from the beging of this io buf*/
    void *offset_to_virt(addr_t offset){
      assert(offset < _n_pages*4096);

      return _allocator.virt_addr(_iob + offset);
    }
    
    Component::io_buffer_t _iob;
    size_t                 _n_pages;
    addr_t                 _phys_addr;

  private:
    static Core::Physical_memory _allocator;
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
  std::map<pid_t, std::vector<IO_memory_instance*>>      _iomem_map;

  KV_ustack_info * _kv_ustack_info; // where ustack will invoke file opers
};


#endif 

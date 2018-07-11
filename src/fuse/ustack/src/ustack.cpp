#include <common/utils.h>
#include <common/str_utils.h>
#include <csignal>
#include <future>
#include "ustack.h"
#include "protocol_generated.h"

using namespace Component;

static IBlock_device * create_nvme_block_device(const std::string device_name);
static IBlock_device * create_posix_block_device(const std::string path);

Ustack::Ustack(const std::string endpoint) : IPC_server(endpoint)
{

  _ipc_thread = new std::thread([=]() { ipc_start(); });
}

Ustack::~Ustack()
{
  TRACE();
  _shutdown = true;
  signal_exit();
  _ipc_thread->join();
  delete _ipc_thread;

  for(auto& i: _channel_map) {
    auto channel = i.second;
    channel->_channel->set_shutdown();
    channel->_channel->unblock_threads();
  }

  for(auto& t_vector: _threads) {
    for(auto& t : t_vector.second) {
      t->join();
    }
  }

  for(auto& i: _channel_map) {
    auto channel = i.second;
    delete channel->_channel;
    delete channel;
  }

  for(auto& i: _shmem_map) {
    for(auto& shmem: i.second)
      delete shmem;
  }

  for(auto& i: _iomem_map) {
    for(auto& iomem: i.second)
      delete iomem;
  }

  store->release_ref();
  block->release_ref();
}

int Ustack::process_message(void* msg,
                            size_t msg_len,
                            void* reply,
                            size_t reply_len)
{
  using namespace Protocol;

  const Message * pmsg = Protocol::GetMessage(msg);
  auto sender_id = pmsg->sender_id();
  PLOG("sender_id: %lu", sender_id);

  /* message: MemoryRequest */
  switch(pmsg->type())
    {
    case MessageType_Memory_request:
      {
        assert(pmsg->element_type() == Element_ElementMemoryRequest);
        PLOG("proto: MemoryRequest {size: %lu}", pmsg->element_as_ElementMemoryRequest()->size());
        auto& shmem_v = _shmem_map[sender_id];
        size_t size_in_pages = round_up(msg_len, PAGE_SIZE) / PAGE_SIZE;
        std::stringstream ss;
        ss << "shm-" << sender_id << "-" << shmem_v.size();

        flatbuffers::FlatBufferBuilder fbb(1024);
        auto response = CreateMessage(fbb,
                                      MessageType_Memory_reply,
                                      getpid(),
                                      Element_ElementMemoryReply,
                                      CreateElementMemoryReply(fbb,
                                                               size_in_pages,
                                                               fbb.CreateString(ss.str())).Union());
        FinishMessageBuffer(fbb, response);    
        memcpy(reply, fbb.GetBufferPointer(), fbb.GetSize());

        /* set up pending shared memory instantiation */
        _pending_shmem.push_back(new Shared_memory_instance(ss.str(), size_in_pages, sender_id));
        break;
      }
    case MessageType_Channel_request:
      {
        PLOG("proto: ChannelRequest");
        std::stringstream ss;
        ss << "channel-" << sender_id;

        flatbuffers::FlatBufferBuilder fbb(1024);
        auto response = CreateMessage(fbb,
                                      MessageType_Channel_reply,
                                      getpid(),
                                      Element_ElementChannelReply,
                                      CreateElementChannelReply(fbb,
                                                                MAX_MESSAGE_SIZE,
                                                                fbb.CreateString(ss.str())).Union());
                                                                
        FinishMessageBuffer(fbb, response);    
        memcpy(reply, fbb.GetBufferPointer(), fbb.GetSize());

        /* defer creation of channel */
        _pending_channels.push_back(new Channel_instance(ss.str(), sender_id));
        break;
      }
    case MessageType_IO_buffer_request:
      {
        assert(pmsg->element_type() == Element_ElementIOBufferRequest);
        size_t n_pages = pmsg->element_as_ElementIOBufferRequest()->n_pages();
        auto& iomem_list = _iomem_map[sender_id];
        auto iomem = new IO_memory_instance(n_pages);
        iomem_list.push_back(iomem);

        flatbuffers::FlatBufferBuilder fbb(1024);
        auto response = CreateMessage(fbb,
                                      MessageType_IO_buffer_reply,
                                      getpid(),
                                      Element_ElementIOBufferReply,
                                      CreateElementIOBufferReply(fbb, iomem->_phys_addr).Union());

        FinishMessageBuffer(fbb, response);    
        memcpy(reply, fbb.GetBufferPointer(), fbb.GetSize());
        break;
      }      
    case MessageType_Shutdown:
      {
        release_resources(sender_id);
        break;
      }
    default:
      throw General_exception("unhandled message received");
    };


  return 0;
}

void Ustack::release_resources(pid_t client_id)
{
  TRACE();

  /* release channel threads */
  if(_channel_map.find(client_id) != _channel_map.end()) {
    auto channel = _channel_map[client_id];
    channel->_channel->set_shutdown();
    channel->_channel->unblock_threads();
  }

  /* release threads */
  if(_threads.find(client_id) != _threads.end()) {
    for(auto& t : _threads[client_id]) {
      t->join();      
      delete t;
      PLOG("Ustack: deleted thread %p", t);
    }
  }
  _threads.erase(client_id);

  /* release channels */
  if(_channel_map.find(client_id) != _channel_map.end()) {
    auto channel = _channel_map[client_id];
    channel->_channel->set_shutdown();
    channel->_channel->unblock_threads();
    delete channel;
    PLOG("Ustack: deleted channel (%p)", channel->_channel);
  }
  _channel_map.erase(client_id);

  /* release shmem */
  if(_shmem_map.find(client_id) != _shmem_map.end()) {
    auto memory_v = _shmem_map[client_id];
    for(auto& s : memory_v) {
      delete s;
      PLOG("Ustack: deleted shmem (%p)", s->_shmem);
    }
  }
  _shmem_map.erase(client_id);

  /* release iomem */
  if(_iomem_map.find(client_id) != _iomem_map.end()) {
    auto memory_v = _iomem_map[client_id];
    for(auto& s : memory_v) {
      delete s;
      PLOG("Ustack: deleted iomem (%p)", s);
    }
  }
  _iomem_map.erase(client_id);

  

  //  for(auto& t_vector: _threads) {
  //   for(auto& t : t_vector.second) {
  //     t->join();
  //   }
  // }

  // for(auto& i: _channel_map) {
  //   auto channel = i.second;
  //   delete channel->_channel;
  //   delete channel;
  // }


  // for(auto& i: _shmem_map) {
  //   for(auto& shmem: i.second) {
  //     delete shmem->_shmem;
  //     delete shmem;
  //   }
  // }


  // /* unblock channels */
  // for(auto& i: _channel_map) {
  //   auto channel = i.second;
  //   if(channel->_client_id == client_id)
  //     channel->_channel->unblock_threads();
  // }

  // for(auto& i: _channel_map) {
  //   auto channel = i.second;
  //   if(channel->_client_id == client_id) {
  //     PLOG("releasing channel (%lu)", client_id);
  //     channel->_channel->unblock_threads();
  //     delete channel->_channel;
  //     delete channel;
  //     _channel_map.erase(client_id);
  //   }
  // }

  // for(auto& t: _threads) {    
  //   t->join();
  // }

  // for(auto& i: _shmem_map) {
  //   auto& v = i.second;
  //   for(std::vector<Shared_memory_instance *>::iterator j=v.begin(); j!=v.end(); j++) {
  //     if((*j)->_client_id == client_id) {
  //       delete (*j)->_shmem;
  //       v.erase(j);
  //     }
  //   }
  // }

}


/*
 * Actual messages are saved in the slab_ring, with in_queue and out_queue saving the refernces 
 */
void Ustack::post_reply(void * reply_msg)
{
  /* connect pending shared memory segments */
  for(auto& p: _pending_shmem) {
    assert(p->_shmem == nullptr);
    PLOG("ustack: creating shared memory (%s, %lu)", p->_id.c_str(), p->_size);
    p->_shmem = new Core::UIPC::Shared_memory(p->_id, p->_size);
    _shmem_map[p->_client_id].push_back(p);
  }
  _pending_shmem.clear();

  /* connect pending shared memory channels */
  for(auto& c: _pending_channels) {
    assert(c->_channel == nullptr);
    PLOG("ustack: creating channel (%s)", c->_id.c_str());
    c->_channel = new Core::UIPC::Channel(c->_id, MAX_MESSAGE_SIZE, MESSAGE_QUEUE_SIZE);
    _channel_map[c->_client_id] = c;

    /* create thread */
    _threads[c->_client_id].push_back(new std::thread([=]() { uipc_channel_thread_entry(c->_channel); }));
  }
  _pending_channels.clear();
}

void Ustack::uipc_channel_thread_entry(Core::UIPC::Channel * channel)
{
  PLOG("worker (%p) starting", channel);
  while(!_shutdown) {
    struct IO_command * msg = nullptr;
    status_t s = channel->recv((void*&)msg);

    if(channel->shutdown() && s == E_EMPTY)
      break;
    if(!msg) continue;
    
    PMAJOR("recv'ed UIPC msg:status=%d, type=%d (%s)",s, msg->type, msg->data);
    
    assert(msg);
    switch(msg->type){

      case IO_TYPE_WRITE:
        /*write to kvstore*/
        if(S_OK == do_kv_write(msg->fuse_fh, msg->offset, msg->sz_bytes)){
          msg->type = IO_WRITE_OK;
        }
        else 
          msg->type = IO_WRITE_FAIL;
        break;

      case IO_TYPE_READ:
        /*read from kvstore*/
        if(S_OK == do_kv_read(msg->fuse_fh, msg->offset, msg->sz_bytes)){
          msg->type = IO_READ_OK;
        }
        else 
          msg->type = IO_READ_FAIL;
        break;

      default:
        /*wrong io type*/
        msg->type = IO_WRONG_TYPE;
    }
    channel->send(msg);   
  }
  PLOG("worker (%p) exiting", channel);
}



/* static functions */


static IBlock_device * create_nvme_block_device(const std::string device_name)
{
  PLOG("creating nvme block device: %s", device_name.c_str());
  
  IBase * comp = load_component("libcomanche-blknvme.so",
                                block_nvme_factory);
  
  assert(comp);
  IBlock_device_factory * fact = (IBlock_device_factory *)
    comp->query_interface(IBlock_device_factory::iid());
  
  cpu_mask_t cpus;
  cpus.add_core(2);
  
  auto block = fact->create(device_name.c_str(), &cpus);
  assert(block);
  fact->release_ref();

  return block;
}

static IBlock_device * create_posix_block_device(const std::string path)
{
  IBase * comp = load_component("libcomanche-blkposix.so",
                                                      block_posix_factory);
  assert(comp);
  PLOG("Block_device factory loaded OK.");
  
  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());
  std::string config_string;
  config_string = "{\"path\":\"";
  //  config_string += "/dev/nvme0n1";1
  config_string += path; //"./blockfile.dat";
  //  config_string += "\"}";
  config_string += "\",\"size_in_blocks\":200000}";

  auto block = fact->create(config_string);
  assert(block);
  fact->release_ref();
  return block;
}

// static members
Core::Physical_memory Ustack::IO_memory_instance::_allocator;

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
  /* block device */
#ifdef USE_NVME_DEVICE
  block = create_nvme_block_device("01:00.0");
#else
  block = create_posix_block_device("./block.dat");
#endif

  /* create blob store */
  assert(block);

  IBase * comp = load_component("libcomanche-blob.so",
                                Component::blob_factory);
  assert(comp);
  IBlob_factory* fact = (IBlob_factory *) comp->query_interface(IBlob_factory::iid());
  assert(fact);

  store = fact->open("cocotheclown",
                     "mydb",
                     block,
                     IBlob_factory::FLAGS_FORMAT); //IBlob_factory::FLAGS_FORMAT); /* pass in lower-level block device */
  
  fact->release_ref();
  
  PINF("Blob component loaded OK.");

  /* test insertion */
  IBlob::blob_t b = store->create("fio.blob",
                                  "dwaddington",
                                  ".jelly",
                                  MB(32));

  _ipc_thread = new std::thread([=]() { ipc_start(); });
}

Ustack::~Ustack()
{
  _shutdown = true;
  signal_exit();
  _ipc_thread->join();
  delete _ipc_thread;

  for(auto& i: _channel_map) {
    auto channel = i.second;
    channel->_channel->unblock_threads();
    delete channel->_channel;
  }

  for(auto& t: _threads) {    
    t->join();
  }

  for(auto& i: _shmem_map)
    for(auto& p: i.second)
      delete p;

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
        PLOG("returning from message loop");
        return 0;
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
        return 0;
      }
    };


  return 0;
}


void Ustack::post_reply(void * reply_msg)
{
  /* connect pending shared memory segments */
  for(auto& p: _pending_shmem) {
    assert(p->_shmem == nullptr);
    PLOG("ustack: creating shared memory (%s, %lu)", p->_id.c_str(), p->_size);
    p->_shmem = new Core::UIPC::Shared_memory(p->_id, p->_size);
    _shmem_map[p->_client_id].push_back(p);
  }

  /* connect pending shared memory channels */
  for(auto& c: _pending_channels) {
    assert(c->_channel == nullptr);
    PLOG("ustack: creating channel (%s)", c->_id.c_str());
    c->_channel = new Core::UIPC::Channel(c->_id, MAX_MESSAGE_SIZE, MESSAGE_QUEUE_SIZE);
    _channel_map[c->_client_id] = c;

    /* create thread */
    _threads.push_back(new std::thread([=]() { uipc_channel_thread_entry(c->_channel); }));
  }
}

void Ustack::uipc_channel_thread_entry(Core::UIPC::Channel * channel)
{
  PLOG("worker (%p) starting", channel);
  while(!_shutdown) {
    struct IO_command * msg = nullptr;
    status_t s = channel->recv((void*&)msg);
    PLOG("recv'ed UIPC msg:status=%d, type=%d (%s)",s, msg->type, msg->data);

    if(s == E_EMPTY && _shutdown) break;
    
    assert(msg);
    msg->type = 101;
    channel->send(msg);
    
    channel->free_msg(msg);
  }
  PLOG("worker (%p) exiting", channel);
}

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
  config_string += "\",\"size_in_blocks\":20000}";

  auto block = fact->create(config_string);
  assert(block);
  fact->release_ref();
  return block;
}



#include <common/utils.h>
#include <common/str_utils.h>
#include <csignal>
#include "ustack.h"

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
  signal_exit();
  _ipc_thread->join();
  delete _ipc_thread;
  
  store->release_ref();
  block->release_ref();
}

int Ustack::process_message(void* msg,
                            size_t msg_len,
                            void* reply,
                            size_t reply_len)
{
  PNOTICE("Got message!!!");
  return 0;
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



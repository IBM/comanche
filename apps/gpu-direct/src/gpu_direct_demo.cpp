#include <boost/program_options.hpp> 
#include <common/exceptions.h>
#include <common/logging.h>
#include <common/dump_utils.h>
#include <core/dpdk.h>
#include <api/components.h>
#include <api/block_itf.h>
#include <iostream>
#include <string>
#include <vector>
#include <stdio.h>
#include <gdrapi.h>
#include <linux/cuda.h>
#include <api/components.h>
#include <api/kvstore_itf.h>
#include <config_comanche.h>

using namespace std;

struct {
  std::string dawn_server;
} Options;

extern "C" void run_cuda(Component::IKVStore * store);

DECLARE_STATIC_COMPONENT_UUID(dawn_client, 0x2f666078,0xcb8a,0x4724,0xa454,0xd1,0xd8,0x8d,0xe2,0xdb,0x87);
DECLARE_STATIC_COMPONENT_UUID(dawn_client_factory, 0xfac66078,0xcb8a,0x4724,0xa454,0xd1,0xd8,0x8d,0xe2,0xdb,0x87);


Component::IKVStore * create_store(const std::string& addr,
                                   const std::string& device,
                                   const unsigned debug_level) {
  using namespace Component;

  std::string path = CONF_COMANCHE_INSTALL;
  path += "/lib/libcomanche-dawn-client.so";
  
  IBase * comp = load_component(path.c_str(), dawn_client_factory);
  assert(comp);
  IKVStore_factory * fact = (IKVStore_factory *) comp->query_interface(IKVStore_factory::iid());

  IKVStore * inst = nullptr;

  inst = fact->create(debug_level,
                      "dwaddington",
                      addr.c_str(),
                      device.c_str());

  fact->release_ref();
  return inst;
}

int main(int argc, char * argv[])
{
  try {
    namespace po = boost::program_options; 
    po::options_description desc("Options"); 
    desc.add_options()
      ("dawn-server", po::value<std::string>()->default_value("10.0.0.22:11911"))
      ("debug", po::value<unsigned>()->default_value(0))
      ("device", po::value<std::string>()->default_value("mlx5_0"))
      ("help", "Show this help")
      ;
 
    po::variables_map vm; 
    po::store(po::parse_command_line(argc, argv, desc),  vm);

    if(vm.count("help")) {
      std::cout << desc;
      return -1;
    }

    auto kvstore = create_store(vm["dawn-server"].as<std::string>(),
                                vm["device"].as<std::string>(),
                                vm["debug"].as<unsigned>());
    
    run_cuda(kvstore);

    kvstore->release_ref();
  }
  catch(boost::program_options::unknown_option err) {
    PERR("unknown option");
    return -1;
  }
  
  return 0;
}


// void Main::run()
// {
//   /* write some data onto device for testing */
//   Component::io_buffer_t iob = _block->allocate_io_buffer(KB(4),KB(4),Component::NUMA_NODE_ANY);
//   void * vaddr = _block->virt_addr(iob);
  
//   memset(vaddr, 0xf, KB(4));
//   sleep(1);
//   _block->write(iob, 0, 1, 1, 0 /* qid */);

//   memset(vaddr, 0, KB(4));
//   _block->read(iob, 0, 1, 1, 0 /* qid */);
//   hexdump(vaddr, 32);
  
//   cuda_run_test(_block);
  

//   // PMAJOR("Calling cuda code ...");

//   // void * vptr = _block->virt_addr(iob);
//   // memset(vptr, 'e', len);
//   // cuda_run_test(vptr, len);

//   // _block->free_io_buffer(iob);
//   // PMAJOR("Closing in 3 seconds ...");
//   // sleep(3);
// }

// void Main::create_block_component(std::string pci_id)
// {
//   using namespace Component;
//   IBase * comp = load_component("libcomanche-blknvme.so", block_nvme_factory);
//   assert(comp);
//   IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());

//   cpu_mask_t m;
//   for(unsigned i=0;i<Options.n_io_threads;i++) {
//     m.add_core(START_IO_CORE + i);
//   }
  
//   _block = fact->create(pci_id.c_str(), &m, nullptr);
//   fact->release_ref();
// }


#include <boost/program_options.hpp> 
#include <common/exceptions.h>
#include <common/logging.h>
#include <common/dump_utils.h>
#include <core/dpdk.h>
#include <api/components.h>
#include <api/block_itf.h>
#include <string>
#include <vector>
#include <stdio.h>
#include <gdrapi.h>
#include <cuda.h>

using namespace std;

#define START_IO_CORE 2

struct {
  unsigned n_client_threads;
  unsigned n_io_threads = 1;
} Options;

extern "C" void cuda_run_test(Component::IBlock_device * block_device);

class Main
{
public:
  Main(const string& pci_id_vector);
  ~Main();

  void run();
  
private:
  void create_block_component(const string vs);

  Component::IBlock_device * _block;
};


Main::Main(const std::string& pci_id_vector)
{
  DPDK::eal_init(64);

  create_block_component(pci_id_vector);
}

Main::~Main()
{
  _block->release_ref();
}


int main(int argc, char * argv[])
{
  Main * m = nullptr;
  
  try {
    namespace po = boost::program_options; 
    po::options_description desc("Options"); 
    desc.add_options()
      ("pci", po::value< string >(), "PCIe id for NVMe drive")
      ;
 
    po::variables_map vm; 
    po::store(po::parse_command_line(argc, argv, desc),  vm);

    if(vm.count("pci")) {
      m = new Main(vm["pci"].as<std::string>());
      m->run();
      delete m;
    }
    else {
      printf("cuda-dma [--pci 8b:00.0 ]\n");
      return -1;
    }
  }
  catch(...) {
    PERR("unhandled exception!");
    return -1;
  }

  return 0;
}

void Main::run()
{
  /* write some data onto device for testing */
  Component::io_buffer_t iob = _block->allocate_io_buffer(KB(4),KB(4),Component::NUMA_NODE_ANY);
  void * vaddr = _block->virt_addr(iob);
  
  memset(vaddr, 0xf, KB(4));
  sleep(1);
  _block->write(iob, 0, 1, 1, 0 /* qid */);

  memset(vaddr, 0, KB(4));
  _block->read(iob, 0, 1, 1, 0 /* qid */);
  hexdump(vaddr, 32);
  
  cuda_run_test(_block);
  

  // PMAJOR("Calling cuda code ...");

  // void * vptr = _block->virt_addr(iob);
  // memset(vptr, 'e', len);
  // cuda_run_test(vptr, len);

  // _block->free_io_buffer(iob);
  // PMAJOR("Closing in 3 seconds ...");
  // sleep(3);
}

void Main::create_block_component(std::string pci_id)
{
  using namespace Component;
  IBase * comp = load_component("libcomanche-blknvme.so", block_nvme_factory);
  assert(comp);
  IBlock_device_factory * fact = (IBlock_device_factory *) comp->query_interface(IBlock_device_factory::iid());

  cpu_mask_t m;
  for(unsigned i=0;i<Options.n_io_threads;i++) {
    m.add_core(START_IO_CORE + i);
  }
  
  _block = fact->create(pci_id.c_str(), &m, nullptr);
  fact->release_ref();
}


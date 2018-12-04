#include <boost/program_options.hpp>
#include <iostream>

#include <common/logging.h>
#include "shard.h"

#define DEFAULT_PMEM_DIR "/mnt/pmem0/"

Program_options g_options;

int main(int argc, char * argv[])
{
  namespace po = boost::program_options;
  
  try {
    po::options_description desc("Options");
    desc.add_options()
      ("help", "Show help")
      ("core", po::value<unsigned>()->default_value(0), "Core id")
      ("debug", po::value<unsigned>()->default_value(0), "Debug level 0-3")
      ("port", po::value<unsigned>()->default_value(11911), "Network port")
      ("data-dir", po::value<std::string>()->default_value(DEFAULT_PMEM_DIR), "Data directory")
      ("forced-exit", "Forced exit")
      ("device", po::value<std::string>()->default_value("mlx5_0"), "Network device (e.g., mlx5_0)")
      ("fabric-provider", po::value<std::string>()->default_value("verbs"), "Fabric provider")
      ("backend", po::value<std::string>()->default_value("mapstore"), "Back-end component")
      ("pci-addr", po::value<std::string>(), "Target PCI address (nvmestore)")
      ("devdax", "Use device dax")
      ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc),  vm);

    if(vm.count("help") > 0) {
      std::cout << desc;
      return -1;
    }

    if(vm.count("pci-addr") > 0)
      g_options.pci_addr = vm["pci-addr"].as<std::string>();
    
    g_options.data_dir = vm["data-dir"].as<std::string>();
    g_options.fabric_provider = vm["fabric-provider"].as<std::string>();
    g_options.device = vm["device"].as<std::string>();
    g_options.backend = vm["backend"].as<std::string>();    
    g_options.core = vm["core"].as<unsigned>();
    g_options.port = vm["port"].as<unsigned>();
    g_options.devdax = vm.count("devdax");

    Dawn::Global::debug_level = g_options.debug_level = vm["debug"].as<unsigned>();

    bool forced_exit = vm.count("forced-exit");

    /* instantiate one shard for the moment */
    Dawn::Shard s(g_options, forced_exit);

    while(!s.exited()) sleep(1);
  }
  catch(po::error e) {
    std::cerr << e.what();
    PLOG("bad command line option configuration");
    return -1;
  }

  return 0;
}

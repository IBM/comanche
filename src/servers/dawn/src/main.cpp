#include <boost/program_options.hpp>
#include <iostream>

#include <common/logging.h>
#include "dawn_config.h"
#include "launcher.h"

#define DEFAULT_PMEM_DIR "/dev/"

Program_options g_options;

int main(int argc, char* argv[]) {
  namespace po = boost::program_options;

  try {
    po::options_description desc("Options");

    desc.add_options()("help", "Show help")                                                        //
        ("config", po::value<std::string>(), "Configuration file")                                 //
        ("debug", po::value<unsigned>()->default_value(0), "Debug level 0-3")                      //
        ("data-dir", po::value<std::string>()->default_value(DEFAULT_PMEM_DIR), "Data directory")  //
        ("forced-exit", "Forced exit") //
        ("device", po::value<std::string>()->default_value("mlx5_0"),"Network device (e.g., mlx5_0)") //
        ("backend", po::value<std::string>()->default_value("mapstore"), "Back-end component")    //
        ("pci-addr", po::value<std::string>(), "Target PCI address (nvmestore)");                 //

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help") > 0) {
      std::cout << desc;
      return -1;
    }

    if (vm.count("config") == 0) {
      std::cout << "--config option is required\n";
      return -1;
    }

    if (vm.count("pci-addr") > 0)
      g_options.pci_addr = vm["pci-addr"].as<std::string>();

    g_options.config_file = vm["config"].as<std::string>();
    g_options.device = vm["device"].as<std::string>();
    g_options.backend = vm["backend"].as<std::string>();
    g_options.forced_exit = vm.count("forced-exit");

    Dawn::Global::debug_level = g_options.debug_level =
        vm["debug"].as<unsigned>();

    /* launch shards */
    {
      Dawn::Shard_launcher launcher(g_options);
      launcher.wait_for_all();
    }
  }
  catch (po::error e) {
    std::cerr << e.what();
    PLOG("bad command line option configuration");
    return -1;
  }

  return 0;
}

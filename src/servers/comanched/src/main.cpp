/*
   Copyright [2017-2019] [IBM Corporation]
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
#include <boost/program_options.hpp> 
#include <iostream>

#include <common/logging.h>
#include <core/postbox.h>
#include <core/uipc.h>
#include <core/dpdk.h>
#include <core/ipc.h>

#include "protocol_generated.h"
#include "service.pb.h"
#include "service.h"

#include <common/mpmc_bounded_queue.h>



int main(int argc, char * argv[])
{
  try {
    namespace po = boost::program_options; 
    po::options_description desc("Options"); 
    desc.add_options()
      ("help", "Show help")
      ("endpoint", po::value<std::string>(), "Endpoint name (UIPC)")
      // ("pci", po::value<std::string>(), "PCIe id for NVMe storage (use raw device)")
      // ("filename", po::value<std::string>(), "POSIX filename for file storage (fixed block)")
      // ("dataset", po::value<std::string>(), "PCIe id for NVMe storage (use raw device)")
      // ("wipe", "Wipe clean data")
      // ("show", "Show existing data summary")
      // ("load", po::value<std::string>(), "Load data from FASTA file list")
      ;
 
    po::variables_map vm; 
    po::store(po::parse_command_line(argc, argv, desc),  vm);

    if(vm.count("help") || vm.count("endpoint")==0 )
      std::cout << desc;

    {
      Service svc(vm["endpoint"].as<std::string>());
      svc.ipc_start();
    }
  }
  catch(...) {
    return -1;
  }



  return 0;
}

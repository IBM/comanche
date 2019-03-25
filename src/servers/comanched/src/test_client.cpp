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

#include <core/ipc.h>
#include "service.pb.h"

class Service_client : Core::IPC_client
{
public:
  Service_client(const std::string& instance_id) :
    Core::IPC_client(std::string("ipc:///tmp//ipc_memory_") + instance_id)
  {
  }

  void allocate_dma_memory(size_t size)
  {
    using namespace ipc_protocol;

    AllocDmaMemory msg;
  }

  void session_request()
  {
    using namespace ipc_protocol;
    
    /* create session */
    CreateSessionRequest msg;
    msg.mutable_hdr()->set_type(ipc_protocol::MessageType::__CreateSessionRequest);
    msg.mutable_hdr()->set_auth_id("authid"); //_connection.auth_id());
    msg.mutable_hdr()->set_auth_token("authtok");
    msg.set_session_type(SessionType::SharedIOSlab);
    msg.set_slab_id("myslab");
    msg.set_memory_size(33);
    msg.set_storage_size(1);
    msg.set_virtual_size(2);
    std::string out_str;
    size_t      reply_msg_len = 0;
    assert(msg.SerializeToString(&out_str));

    void *reply_msg = send_and_wait(out_str.c_str(),
                                    out_str.length(),
                                    &reply_msg_len);

    CreateSessionReply reply;
    reply.ParseFromArray(reply_msg, reply_msg_len);

    if (reply.status() != 0) {
      throw Constructor_exception("session creation failed.");
    }

  }
};
 
int main(int argc, char * argv[])
{
  try {
    namespace po = boost::program_options; 
    po::options_description desc("Options"); 
    desc.add_options()
      ("help", "Show help")
      ("endpoint", po::value<std::string>(), "Endpoint name (IPC)")
      ;
 
    po::variables_map vm; 
    po::store(po::parse_command_line(argc, argv, desc),  vm);

    if(vm.count("help") || vm.count("endpoint")==0 )
      std::cout << desc;

    {
      
      Service_client svc(vm["endpoint"].as<std::string>());

      for(unsigned i=0;i<5;i++) {
        sleep(1);
        svc.session_request();
      }
    }
  }
  catch(...) {
    return -1;
  }


  
  return 0;
}

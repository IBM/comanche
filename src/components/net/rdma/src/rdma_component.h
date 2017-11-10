/*
   Copyright [2017] [IBM Corporation]

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

#ifndef __RDMA_COMPONENT_H__
#define __RDMA_COMPONENT_H__

#include <component/base.h>
#include <api/rdma_itf.h>


class Rdma_component : public Component::IRdma
{  
private:
  static constexpr bool option_DEBUG = true;

public:
  /** 
   * Constructor
   * 
   * @param block_device Block device interface
   * 
   */
  Rdma_component(const std::string& device_name);

  /** 
   * Destructor
   * 
   */
  virtual ~Rdma_component();

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0x7b2b8cb7,0x5747,0x40e9,0x915d,0x69,0x43,0xa8,0xc5,0x9a,0x60);
  
  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IRdma::iid()) {
      return (void *) static_cast<Component::IRdma*>(this);
    }
    else return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

public:

  virtual status_t connect(const std::string& peer_name, int port) override;
  virtual status_t disconnect() override;

private:

};


class Rdma_component_factory : public Component::IRdma_factory
{  
public:

  /** 
   * Component/interface management
   * 
   */
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xfacb8cb7,0x5747,0x40e9,0x915d,0x69,0x43,0xa8,0xc5,0x9a,0x60);
  
  void * query_interface(Component::uuid_t& itf_uuid) override {
    if(itf_uuid == Component::IRdma_factory::iid()) {
      return (void *) static_cast<Component::IRdma_factory*>(this);
    }
    else return NULL; // we don't support this interface
  }

  void unload() override {
    delete this;
  }

  virtual Component::IRdma * create(const std::string& device_name) override
  {    
    Component::IRdma * obj = static_cast<Component::IRdma*>(new Rdma_component(device_name));    
    obj->add_ref();
    return obj;
  }

};



#endif

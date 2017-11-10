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

#ifndef __API_RDMA_ITF__
#define __API_RDMA_ITF__

#include <common/exceptions.h>
#include <api/rdma_itf.h>

namespace Component
{

class IRdma : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0xfbf7b335,0x9309,0x4f6b,0x8b44,0x92,0x46,0x8b,0xb5,0x6f,0x31);

  virtual status_t connect(const std::string& peer_name, int port) = 0;
  virtual status_t disconnect() = 0;
};

class IRdma_factory : public Component::IBasek
{
public:
  DECLARE_INTERFACE_UUID(0xfac7b335,0x9309,0x4f6b,0x8b44,0x92,0x46,0x8b,0xb5,0x6f,0x31);

  virtual IRdma * create(const std::string& device_name) = 0;
};

} // Component

#endif // __API_RDMA_ITF__

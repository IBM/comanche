/*
   Copyright [2018] [IBM Corporation]

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

#ifndef __API_CLUSTER_ITF__
#define __API_CLUSTER_ITF__

#include <functional>
#include <common/exceptions.h>
#include <component/base.h>

namespace Component
{

/** 
 * Local IP-based clustering
 * 
 */
class ICluster : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0x9e9f55e3,0xf056,0x4273,0xa9c2,0x63,0xab,0x28,0xaa,0xa5,0x81);

  /** 
   * Start node
   * 
   */
  virtual void start_node() = 0;

  /** 
   * Stop node; signal peers gracefully
   * 
   */
  virtual void stop_node() = 0;

  /** 
   * Create/join a group
   * 
   * @param group Name of group
   */
  virtual void group_join(const std::string& group) = 0;

  /** 
   * Leave a group
   * 
   * @param group Name of group
   */
  virtual void group_leave(const std::string& group) = 0;

  /** 
   * Broadcast message to group
   * 
   * @param group Group to send to
   * @param type Message type designator
   * @param message Message to send
   */
  virtual void shout(const std::string& group,
                     const std::string& type,
                     const std::string& message) = 0;

  /** 
   * Send message to single peer
   * 
   * @param peer_uuid Peer UUID
   * @param type Message type designator
   * @param message Message to send
   */
  virtual void whisper(const std::string& peer_uuid,
                       const std::string& type,
                       const std::string& message) = 0;

  /** 
   * Poll for incoming messages
   * 
   * @param callback Callback
   */
  virtual void poll_recv(std::function<void(const std::string& sender_uuid,
                                            const std::string& type,
                                            const std::string& message)> callback) = 0;

  /** 
   * Return the node name
   * 
   * 
   * @return 
   */
  virtual std::string node_name() const = 0;
  
  /** 
   * Return cluster-wide unique identifier for this node
   * 
   * 
   * @return UUID of the node
   */
  virtual std::string uuid() const = 0;

  /** 
   * Dump debugging information
   * 
   */
  virtual void dump_info() const = 0;
};


class ICluster_factory : public Component::IBase
{
public:
  DECLARE_INTERFACE_UUID(0xfacf55e3,0xf056,0x4273,0xa9c2,0x63,0xab,0x28,0xaa,0xa5,0x81);

  /** 
   * Create an instance of the Cluster component
   * 
   * @param device_name Device name (e.g., mlnx5_0)
   * 
   * @return Pointer to new instance
   */
  virtual ICluster * create(const std::string& node_name,
                            const std::string& end_point) = 0;

  virtual ICluster * create(const std::string& node_name) = 0;

};

} // Component

#endif // __API_RDMA_ITF__

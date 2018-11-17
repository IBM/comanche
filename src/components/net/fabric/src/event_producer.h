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

#ifndef _EVENT_PRODUCER_H_
#define _EVENT_PRODUCER_H_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"
#include <rdma/fabric.h> /* fid_t */
#pragma GCC diagnostic pop
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wshadow"
#include <rdma/fi_domain.h> /* fi_eq_cm_entry */
#pragma GCC diagnostic pop

#include <cstdint>
#include <tuple>

class event_consumer;

class event_producer
{
protected:
  ~event_producer() {}
public:
  virtual void register_pep(::fid_t ep, event_consumer &ec) = 0;
  virtual void register_aep(::fid_t ep, event_consumer &ec) = 0;
  virtual void deregister_endpoint(::fid_t ep) = 0;
  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_ep_bind fail
   */
  virtual void bind(::fid_ep &ep) = 0;
  /*
   * @throw fabric_runtime_error : std::runtime_error : ::fi_pep_bind fail
   */
  virtual void bind(::fid_pep &ep) = 0;
  /* file descriptor for detecting activity on the event queue */
  virtual int fd() const = 0;
  /**
   * read functions for the event queue
   *
   * @throw fabric_bad_alloc : std::bad_alloc - libfabric out of memory (creating a new server)
   * @throw std::system_error : pselect fail
   */
  virtual void read_eq() = 0;
  /*
   * NOTE: the wait is edge-triggered, so a wait on an fd which already has an event
   * will time out.
   *
   * @throw fabric_runtime_error : std::runtime_error : ::fi_control fail
   */
  virtual void wait_eq() = 0;
};

#endif

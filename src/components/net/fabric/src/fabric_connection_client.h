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

#ifndef _FABRIC_CONNECTION_CLIENT_H_
#define _FABRIC_CONNECTION_CLIENT_H_

#include "fabric_op_control.h"

#include <cstdint> /* uint16_t */
#include <string>

struct fi_info;

class event_producer;
class Fabric;

class Fabric_connection_client
  : public Fabric_op_control
{
  event_producer &_ev;
  /* BEGIN Fabric_op_control */
  void solicit_event() const override;
  void wait_event() const override;
  /* END Fabric_op_control */
  void expect_event_sync(std::uint32_t event_exp) const;
public:
  explicit Fabric_connection_client(
    Fabric &fabric
    , event_producer &ep
    , ::fi_info & info
    , const std::string & remote
    , std::uint16_t control_port
  );
  ~Fabric_connection_client();
};

#endif

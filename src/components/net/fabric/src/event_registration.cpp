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

/*
 * Authors:
 *
 */

#include "event_registration.h"

#include "event_producer.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wshadow"
#include <rdma/fi_endpoint.h> /* fid_ep */
#pragma GCC diagnostic pop

#include <iostream> /* cerr */

event_registration::event_registration(event_producer &ev_, event_consumer &ec_, ::fid_ep &ep_)
  : _ev(ev_)
  , _ep(&ep_.fid)
{
  _ev.register_aep(_ep, ec_);
  _ev.bind(ep_);
}

event_registration::event_registration(event_producer &ev_, event_consumer &ec_, ::fid_pep &ep_)
  : _ev(ev_)
  , _ep(&ep_.fid)
{
  _ev.register_pep(_ep, ec_);
  _ev.bind(ep_);
}

event_registration::~event_registration()
try
{
  _ev.deregister_endpoint(_ep);
}
catch ( const std::exception &e )
{
  std::cerr << __func__ << " exception " << e.what() << "\n";
}

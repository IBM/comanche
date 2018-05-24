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

#ifndef _FABRIC_ASYNC_REQ_RECORD_H_
#define _FABRIC_ASYNC_REQ_RECORD_H_

class Fabric_comm;

/*
 * Additional layer of context for each async operation, to direct the
 * completion of async requests issued by fabric_comm. The user-level
 * context is embedded, and the fabric_comm pointer is added to
 * remember the fabric_comm to be used for completion.
 */
class async_req_record
{
  Fabric_comm *_comm;
  void *_context;
public:
  explicit async_req_record(Fabric_comm *comm_, void *context_)
    : _comm(comm_)
    , _context(context_)
  {
  }
  Fabric_comm *comm() const { return _comm; }
  void *context() const { return _context; }
};

#endif

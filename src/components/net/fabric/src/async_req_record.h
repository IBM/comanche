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


/*
 * Authors:
 *
 */

#ifndef _FABRIC_ASYNC_REQ_RECORD_H_
#define _FABRIC_ASYNC_REQ_RECORD_H_

class Fabric_cq_grouped;

/*
 * Additional layer of context for each async operation, to direct the
 * completion of async requests issued by fabric_comm. The user-level
 * context is embedded, and the fabric_cq pointer is added to
 * remember the fabric_cq to be used for completion.
 */
class async_req_record
{
  Fabric_cq_grouped *_cq;
  void *_context;
public:
  explicit async_req_record(Fabric_cq_grouped *cq_, void *context_)
    : _cq(cq_)
    , _context(context_)
  {
  }
  ~async_req_record()
  {
  }
  async_req_record(const async_req_record &) = delete;
  async_req_record &operator=(const async_req_record &) = delete;
  Fabric_cq_grouped *cq() const { return _cq; }
  void *context() const { return _context; }
};

#endif

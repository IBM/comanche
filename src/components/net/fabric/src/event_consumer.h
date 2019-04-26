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


#ifndef _EVENT_CONSUMER_H_
#define _EVENT_CONSUMER_H_

#include <cstdint> /* uint32_t */

struct fi_eq_cm_entry;
struct fi_eq_err_entry;

class event_consumer
{
protected:
  ~event_consumer() {}
public:
  virtual void cb(std::uint32_t event, fi_eq_cm_entry &entry) noexcept = 0;
  virtual void err(fi_eq_err_entry &entry) noexcept = 0;
};

#endif

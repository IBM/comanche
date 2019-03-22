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
#ifndef __YCSB_DB_FACT_H__
#define __YCSB_DB_FACT_H__

#include "aerospikedb.h"
#include "dawndb.h"
#include "db.h"
#include "memcached.h"
#include "properties.h"
#include "redisc.h"

namespace ycsb
{
class DBFactory {
 public:
  static DB* create(Properties& props, unsigned core = 0)
  {
    if (props.getProperty("db") == "dawn") {
      return new DawnDB(props, core);
    }
    else if (props.getProperty("db") == "memcached") {
      return new Memcached(props, core);
    }
    else if (props.getProperty("db") == "aerospike") {
      return new AerospikeDB(props, core);
    }
    else if (props.getProperty("db") == "redis") {
      return new RedisC(props, core);
    }
    else
      return nullptr;
  }
};

}  // namespace ycsb

#endif

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
#ifndef __YCSB_DAWNDB_H__
#define __YCSB_DAWNDB_H__

#include <api/components.h>
#include <api/kvstore_itf.h>
#include "../../kvstore/stopwatch.h"
#include "db.h"
#include "properties.h"

using namespace ycsbutils;

namespace ycsb
{
class DawnDB : public DB {
  public:
   DawnDB(Properties &props, unsigned core = 0);
   virtual ~DawnDB();
   virtual int  get(const string &table,
                    const string &key,
                    char *        value,
                    bool          direct = false) override;
   virtual int  put(const string &table,
                    const string &key,
                    const string &value,
                    bool          direct = false) override;
   virtual int  update(const string &table,
                       const string &key,
                       const string &value,
                       bool          direct = false) override;
   virtual int  erase(const string &table, const string &key) override;
   virtual int  scan(const string &                table,
                     const string &                key,
                     int                           count,
                     vector<pair<string, string>> &results) override;
   virtual void init(Properties &props, unsigned core = 0) override;
   virtual void clean() override;

  private:
   Component::IKVStore *       client;
   Component::IKVStore::pool_t pool;
};

}  // namespace ycsb

#endif

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
#ifndef __YCSB_DB_H__
#define __YCSB_DB_H__

#include <map>
#include <vector>
#include "properties.h"

using namespace ycsbutils;
using namespace std;

namespace ycsb
{
class DB {
 public:
  virtual int  get(const string &table,
                   const string &key,
                   char *        value,
                   bool          direct = false)                              = 0;
  virtual int  put(const string &table,
                   const string &key,
                   const string &value,
                   bool          direct = false)                              = 0;
  virtual int  update(const string &table,
                      const string &key,
                      const string &value,
                      bool          direct = false)                           = 0;
  virtual int  erase(const string &table, const string &key)         = 0;
  virtual int  scan(const string &                table,
                    const string &                key,
                    int                           count,
                    vector<pair<string, string>> &results)           = 0;
  virtual void init(Properties &props, unsigned core = 0)            = 0;
  virtual void clean()                                               = 0;
  virtual ~DB(){};
};
}  // namespace ycsb

#endif

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
//
//  generator.h
//  YCSB-C
//
//  Created by Jinglei Ren on 12/6/14.
//  Copyright (c) 2014 Jinglei Ren <jinglei@ren.systems>.
//

#ifndef YCSB_C_GENERATOR_H_
#define YCSB_C_GENERATOR_H_

#include <cstdint>
#include <string>

namespace ycsbc {
template <typename Value>
class Generator {
 public:
  virtual Value Next() = 0;
  virtual Value Last() = 0;
  virtual ~Generator() { }
};

} // ycsbc

#endif // YCSB_C_GENERATOR_H_

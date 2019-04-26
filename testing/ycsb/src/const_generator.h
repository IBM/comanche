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
//  const_generator.h
//  YCSB-C
//
//  Created by Jinglei Ren on 12/10/14.
//  Copyright (c) 2014 Jinglei Ren <jinglei@ren.systems>.
//

#ifndef YCSB_C_CONST_GENERATOR_H_
#define YCSB_C_CONST_GENERATOR_H_

#include "generator.h"
#include <cstdint>

namespace ycsbc {
class ConstGenerator : public Generator<uint64_t> {
 public:
  ConstGenerator(int constant) : constant_(constant) {}
  uint64_t Next() { return constant_; }
  uint64_t Last() { return constant_; }

 private:
  uint64_t constant_;
};

} // ycsbc

#endif // YCSB_C_CONST_GENERATOR_H_

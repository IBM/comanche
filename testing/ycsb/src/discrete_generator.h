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
//  discrete_generator.h
//  YCSB-C
//
//  Created by Jinglei Ren on 12/6/14.
//  Copyright (c) 2014 Jinglei Ren <jinglei@ren.systems>.
//

#ifndef YCSB_C_DISCRETE_GENERATOR_H_
#define YCSB_C_DISCRETE_GENERATOR_H_

#include "generator.h"

#include <atomic>
#include <cassert>
#include <mutex>
#include <vector>
#include "properties.h"

namespace ycsbc {

template <typename Value>
class DiscreteGenerator : public Generator<Value> {
 public:
  DiscreteGenerator() : sum_(0) { }
  void AddValue(Value value, double weight);

  Value Next();
  Value Last() { return last_; }

 private:
  std::vector<std::pair<Value, double>> values_;
  double sum_;
  std::atomic<Value> last_;
  std::mutex mutex_;
};

template <typename Value>
inline void DiscreteGenerator<Value>::AddValue(Value value, double weight) {
  if (values_.empty()) {
    last_ = value;
  }
  values_.push_back(std::make_pair(value, weight));
  sum_ += weight;
}

template <typename Value>
inline Value DiscreteGenerator<Value>::Next() {
  mutex_.lock();
  double chooser = ycsbutils::RandomDouble();
  mutex_.unlock();
  
  for (auto p = values_.cbegin(); p != values_.cend(); ++p) {
    if (chooser < p->second / sum_) {
      return last_ = p->first;
    }
    chooser -= p->second / sum_;
  }
  
  assert(false);
  return last_;
}

} // ycsbc

#endif // YCSB_C_DISCRETE_GENERATOR_H_

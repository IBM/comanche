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

#ifndef _BAD_ALIGNED_ALLOC_H_
#define _BAD_ALIGNED_ALLOC_H_

#include <cstddef> /* size_t */
#include <new> /* bad_alloc */
#include <string>

class bad_aligned_alloc
  : public std::bad_alloc
{
  std::string _what;
public:
  explicit bad_aligned_alloc(std::size_t sz, long align);
  const char *what() const noexcept;
};

#endif

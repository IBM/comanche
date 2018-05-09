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

#ifndef _SYSTEM_FAIL_H_
#define _SYSTEM_FAIL_H_

#include <system_error>

static inline void system_fail(int e, const std::string &context)
{
    throw std::system_error{std::error_code{e, std::system_category()}, context};
}

#endif

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


#include "fabric_bad_alloc.h"

/*
 * Authors:
 *
 */

#include <stdexcept> /* bad_alloc */
#include <string>

fabric_bad_alloc::fabric_bad_alloc(const std::string &what)
  : std::bad_alloc{}
  , _what{"fabric_bad_alloc: " + what}
{}

const char *fabric_bad_alloc::what() const noexcept
{
  return _what.c_str();
}

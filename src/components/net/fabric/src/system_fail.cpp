/*
   Copyright [2019] [IBM Corporation]

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

#include "system_fail.h"

#include <netdb.h> /* gai_strerror */
#include <system_error> /* error_category */

namespace
{
  class gai_error_category
    : public std::error_category
  {
    const char *name() const noexcept override { return "getaddrinfo"; }
    std::string message(int condition) const override
    {
      return ::gai_strerror(condition);
    }
  };

  gai_error_category gai_errors;
}

const std::error_category& gai_category() noexcept
{
    return gai_errors;
}

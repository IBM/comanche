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


#ifndef _FABRIC_JSON_H_
#define _FABRIC_JSON_H_

#include <memory>
#include <string>

struct fi_info;

/**
 * @throw std::domain_error : json file parse-detected error
 */
std::shared_ptr<fi_info> parse_info(const std::string &s, std::shared_ptr<fi_info> info);
/**
 * @throw std::bad_alloc : fabric_bad_alloc - libfabric out of memory
 * @throw std::domain_error : json file parse-detected error
 */
std::shared_ptr<fi_info> parse_info(const std::string &s);

#endif

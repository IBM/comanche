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


#ifndef _COMANCHE_HSTORE_POBJ_BAD_ALLOC_H
#define _COMANCHE_HSTORE_POBJ_BAD_ALLOC_H

#include <cstring>
#include <stdexcept>
#include <string>

class pobj_bad_alloc
	: public std::bad_alloc
{
	int _err;
	std::string _what;
public:
	/* Note: it would make more senst for count, not size, to vary */
	pobj_bad_alloc(std::size_t pad, std::size_t count, std::size_t size_min, std::size_t size_max, int err_)
		: _err(err_)
		, _what(std::string("pobj_bad_alloc: ") + std::to_string(pad) + "+" + std::to_string(count) + "*" + std::to_string(size_min) + ".." + std::to_string(size_max) + " " + strerror(_err))
	{}
	const char *what() const noexcept override
	{
		return _what.c_str();
	}
};

#endif

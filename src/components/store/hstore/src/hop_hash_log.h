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


#ifndef COMANCHE_HSTORE_HOP_HASH_LOG_H
#define COMANCHE_HSTORE_HOP_HASH_LOG_H

#include <iosfwd>
#include <string>
#include <sstream>

namespace hop_hash_log_impl
{
	void wr(const std::string &s);
}

template <bool>
	class hop_hash_log;

template <>
	class hop_hash_log<true>
	{
		static void wr(std::ostream &)
		{
		}

		template<typename T, typename... Args>
			static void wr(std::ostream &o, const T & e, const Args & ... args)
			{
				o << e;
				wr(o, args...);
			}

	public:
		template<typename... Args>
			static void write(const Args & ... args)
			{
				std::ostringstream o;
				wr(o, args...);
				hop_hash_log_impl::wr(o.str());
			}
	};

template <>
	class hop_hash_log<false>
	{
	public:
		template<typename... Args>
			static void write(const Args & ...)
			{
			}
	};

#endif

/*
   Copyright [2018-2019] [IBM Corporation]
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

#ifndef _HSTORE_HISTOGRAM_LOG2_H_
#define _HSTORE_HISTOGRAM_LOG2_H_

#include <array>
#include <limits>

namespace util
{
	namespace
	{
		template <typename clz_arg_t>
			int clz(clz_arg_t);

		template <>
			int clz(unsigned v)
			{
				/* Note: If no builtin clzi/clz/clzll, see deBruijn sequence method,
				 * or Hacker's Delight section 5-3
				 */
				return __builtin_clz(v);
			}

		template <>
			int clz(unsigned long v)
			{
				return __builtin_clzl(v);
			}

		template <>
			int clz(unsigned long long v)
			{
				return __builtin_clzll(v);
			}
	}

template <typename clz_arg_t>
	class histogram_log2
	{
	public:
		static auto constexpr array_size = std::numeric_limits<clz_arg_t>::digits;
		using array_t = std::array<unsigned, array_size>;
	private:
		array_t _hist;
	public:
		histogram_log2()
			: _hist{}
		{}

		void enter(clz_arg_t v) {
			++_hist[ v ? array_size - clz(v) : 0];
		}

		void remove(clz_arg_t v) {
			--_hist[ v ? array_size - clz(v) : 0];
		}

		const array_t &data() const { return _hist; }
	};
}

#endif


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


#ifndef _COMANCHE_HSTORE_PERISTENT_H
#define _COMANCHE_HSTORE_PERISTENT_H

#include "perishable.h"
#include "test_flags.h" /* TEST_HSTORE_PERISHABLE */
#include <atomic>

template <typename T>
	class persistent
	{
		T _v;
	public:
		persistent()
			: _v((perishable::tick(), T()))
		{
		}
		persistent(const persistent &other)
			: _v((perishable::tick(), other._v))
		{
		}
		persistent(const T &t)
			: _v((perishable::tick(), t))
		{
		}
		persistent<T> &operator=(const persistent &other)
		{
			perishable::tick();
			_v = other._v;
			return *this;
		};
		persistent<T> &operator=(const T &t)
		{
			perishable::tick();
			_v = t;
			return *this;
		};
		persistent<T> &operator&=(const T &t)
		{
			perishable::tick();
			_v &= t;
			return *this;
		};
		persistent<T> &operator|=(const T &t)
		{
			perishable::tick();
			_v |= t;
			return *this;
		};
		operator T() const
		{
			return _v;
		}
		T load() const
		{
			return _v;
		}
		persistent<T> &operator++()
		{
			perishable::tick();
			++_v;
			return *this;
		};
		persistent<T> &operator--()
		{
			perishable::tick();
			--_v;
			return *this;
		};
		auto &operator*() const
		{
			return *_v;
		};
		T operator->() const
		{
			return _v;
		};
		operator bool() const
		{
			return bool(_v);
		};
	};

template <typename T>
	class persistent_atomic
	{
		std::atomic<T> _v;
		static_assert(sizeof _v <= 8, "persistent store is too large");
	public:
		persistent_atomic()
			: _v((perishable::tick(), T{}))
		{
		}
		persistent_atomic(const persistent_atomic &other)
			: _v((perishable::tick(), other._v.load()))
		{
		}
		persistent_atomic(const T &t)
			: _v((perishable::tick(), t))
		{
		}
		persistent_atomic<T> &operator=(const persistent_atomic &other)
		{
			perishable::tick();
			_v = other._v.load();
			return *this;
		};
		persistent_atomic<T> &operator=(const T &t)
		{
			perishable::tick();
			_v = t;
			return *this;
		};
		persistent_atomic<T> &operator+=(const T &t)
		{
			perishable::tick();
			_v += t;
			return *this;
		};
		persistent_atomic<T> &operator-=(const T &t)
		{
			perishable::tick();
			_v -= t;
			return *this;
		};
		persistent_atomic<T> &operator&=(const T &t)
		{
			perishable::tick();
			_v &= t;
			return *this;
		};
		persistent_atomic<T> &operator|=(const T &t)
		{
			perishable::tick();
			_v |= t;
			return *this;
		};
		operator T() const
		{
			return _v;
		}
		persistent_atomic<T> &operator++()
		{
			perishable::tick();
			++_v;
			return *this;
		};
		persistent_atomic<T> &operator--()
		{
			perishable::tick();
			--_v;
			return *this;
		};
		auto &operator*() const
		{
			return *_v;
		};
		T operator->() const
		{
			return _v;
		};
	};

#if TEST_HSTORE_PERISHABLE
template <typename T>
	using persistent_t = persistent<T>;
template <typename T>
	using persistent_atomic_t = persistent_atomic<T>;
#else
template <typename T>
	using persistent_t = T;
template <typename T>
	using persistent_atomic_t = T;
#endif

#endif

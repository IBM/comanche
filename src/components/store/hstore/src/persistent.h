#ifndef _DAWN_HSTORE_PERISTENT_H
#define _DAWN_HSTORE_PERISTENT_H

#include "perishable.h"
#include <atomic>

#if USE_PERISHABLE
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

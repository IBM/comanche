#ifndef _DAWN_HSTORE_DUMMY_SHARED_MUTEX_H_
#define _DAWN_HSTORE_DUMMY_SHARED_MUTEX_H_

namespace dummy
{
	struct shared_mutex
	{
		/* BasicLockable */
		void lock() {}
		void unlock() {}
		/* Lockable */
		bool try_lock() { return true; }
		/* SharedMutex */
		void lock_shared() {}
		bool try_lock_shared() { return true; }
		void unlock_shared() {}
	};
}

#endif

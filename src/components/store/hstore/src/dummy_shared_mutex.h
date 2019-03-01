/*
 * (C) Copyright IBM Corporation 2018. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#ifndef _COMANCHE_HSTORE_DUMMY_SHARED_MUTEX_H_
#define _COMANCHE_HSTORE_DUMMY_SHARED_MUTEX_H_

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

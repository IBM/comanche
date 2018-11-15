/*
 * Hopscotch hash table - template Key, Value, and allocators
 */

/*
 * ===== bucket =====
 */

template <typename Value>
	impl::bucket<Value>::bucket(owner owner_)
		: owner{owner_}
		, content<Value>()
	{}

template <typename Value>
	impl::bucket<Value>::bucket()
		: bucket(owner())
	{}

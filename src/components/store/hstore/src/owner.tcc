/*
 * (C) Copyright IBM Corporation 2018. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

/*
 * Hopscotch hash table - template Key, Value, and allocators
 */

/*
 * ===== owner =====
 */

template <typename Lock>
	auto impl::owner::owned(std::size_t table_size_, Lock &s) const -> std::string
	{
		std::string st = "";
		if ( _pos != pos_undefined )
		{
			auto pos = _pos;
			std::string delim = "";
			for ( auto v = value(s); v; v>>=1, ++pos %= table_size_ )
			{
				if ( v & mask_from_pos(0U) )
				{
					st += delim + std::to_string(pos);
				delim = " ";
				}
			}
		}
		return "(" + st + ")";
	}

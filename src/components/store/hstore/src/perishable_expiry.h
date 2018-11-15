#ifndef _DAWN_HSTORE_PERISHABLE_EXPIRY_H
#define _DAWN_HSTORE_PERISHABLE_EXPIRY_H

#include <stdexcept>

class perishable_expiry
	: public std::runtime_error
{
public:
	perishable_expiry()
		: std::runtime_error("perishable timer expired")
	{}
};

#endif

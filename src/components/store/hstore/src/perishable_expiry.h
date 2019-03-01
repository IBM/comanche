/*
 * (C) Copyright IBM Corporation 2018. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#ifndef _COMANCHE_HSTORE_PERISHABLE_EXPIRY_H
#define _COMANCHE_HSTORE_PERISHABLE_EXPIRY_H

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

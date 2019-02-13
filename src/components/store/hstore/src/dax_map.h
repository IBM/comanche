/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#ifndef COMANCHE_HSTORE_DAX_MAP_H
#define COMANCHE_HSTORE_DAX_MAP_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#pragma GCC diagnostic ignored "-Weffc++"
struct iovec; /* required but not declared by nupm/dax_map.h */
#include <nupm/dax_map.h>
#pragma GCC diagnostic pop

#include <string>

class Devdax_manager
	: public nupm::Devdax_manager
{
public:
	Devdax_manager(
		const std::string &dax_map
		, bool force_reset = false
	);
};

#endif

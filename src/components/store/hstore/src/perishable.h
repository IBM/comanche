/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#ifndef _COMANCHE_HSTORE_PERISHABLE_H
#define _COMANCHE_HSTORE_PERISHABLE_H

#include <cstdint>

class perishable
{
	static bool _enabled;
	static std::uint64_t _initial;
	static std::uint64_t _time_to_live;
public:
	static void reset(std::uint64_t n);
	static void enable(bool e);
	static void tick();
	static void test(); /* like tick, but without the decrement */
	static void report();
};

#endif

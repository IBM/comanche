/*
 * (C) Copyright IBM Corporation 2018, 2019. All rights reserved.
 * US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
 */

#ifndef COMANCHE_HSTORE_PERSISTER_H
#define COMANCHE_HSTORE_PERSISTER_H

#include <cstddef> /* size_t */

/* default "persister" for persistent memory: a no-op.
 */
class persister
{
public:
  void persist(const void *, std::size_t) {}
};

#endif

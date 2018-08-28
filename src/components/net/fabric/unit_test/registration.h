#ifndef _TEST_REGISTRATION_H_
#define _TEST_REGISTRATION_H_

#include "delete_copy.h"

#include <api/fabric_itf.h> /* Fabric_connection, memory_region_t */
#include <cstddef> /* size_t */
#include <cstdint> /* uint64_t */

class registration
{
  Component::IFabric_connection &_cnxn;
  Component::IFabric_connection::memory_region_t _region;
  std::uint64_t _key;
  void *_desc;
  DELETE_COPY(registration);
public:
  explicit registration(Component::IFabric_connection &cnxn_, const void *contig_addr_, std::size_t size_, std::uint64_t key_, std::uint64_t flags_);
  registration(registration &&);
  registration &operator=(registration &&);
  ~registration();

  std::uint64_t key() const { return _key; }
  void *desc() const { return _desc; }
};

#endif

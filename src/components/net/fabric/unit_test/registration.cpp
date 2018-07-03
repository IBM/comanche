#include "registration.h"

#include "eyecatcher.h"
#include <api/fabric_itf.h> /* Fabric_connection */
#include <exception>
#include <iostream> /* cerr */

registration::registration(Component::IFabric_connection &cnxn_, const void *contig_addr_, std::size_t size_, std::uint64_t key_, std::uint64_t flags_)
  : _cnxn(cnxn_)
  , _region(_cnxn.register_memory(contig_addr_, size_, key_, flags_))
  , _key(_cnxn.get_memory_remote_key(_region))
{
}

registration::~registration()
try
{
  _cnxn.deregister_memory(_region);
}
catch ( std::exception &e )
{
  std::cerr << __func__ << " exception " << e.what() << eyecatcher << std::endl;
}

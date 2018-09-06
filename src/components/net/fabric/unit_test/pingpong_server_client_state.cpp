#include "pingpong_server_client_state.h"

#include <api/fabric_itf.h>

client_state::client_state(
  Component::IFabric_server_factory &factory_
  , unsigned iteration_count_
  , std::size_t msg_size_
)
  : sc(factory_)
  , st(
    sc.cnxn()
    , iteration_count_
    , msg_size_
  )
{
}

client_state::~client_state()
{
}


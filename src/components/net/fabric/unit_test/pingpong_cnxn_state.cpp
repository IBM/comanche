#include "pingpong_cnxn_state.h"

#include <utility> /* move */

void cb_ctxt::cb(void *ctxt_, ::status_t stat_, std::uint64_t flags_, std::size_t len_, void *err_)
{
  auto ctxt = static_cast<const cb_ctxt *>(ctxt_);
  /* calls one of send_cb, recv0_cb, recv1_cb */
  return (ctxt->cb_call)(ctxt->state, stat_, flags_, len_, err_);
}

cb_ctxt::cb_ctxt(cnxn_state *state_, cb_t cb_call_)
  : state(state_)
  , cb_call(cb_call_)
{
}

buffer_state::buffer_state(Component::IFabric_connection &cnxn_, std::size_t buffer_size_, std::uint64_t remote_key_, std::size_t msg_size_)
  : _rm{cnxn_, buffer_size_,  remote_key_}
  , v{{&_rm[0], msg_size_}}
  , d{{_rm.desc()}}
{}

cnxn_state::cnxn_state(
  cb_ctxt::cb_t send_cb_
  , cb_ctxt::cb_t recv0_cb_
  , cb_ctxt::cb_t recv1_cb_
  , Component::IFabric_active_endpoint_comm &comm_
  , std::size_t buffer_size_
  , std::uint64_t remote_key_
  , unsigned iteration_count_
  , std::size_t msg_size_
)
  : send_ctxt(this, send_cb_)
  , recv0_ctxt(this, recv0_cb_)
  , recv1_ctxt(this, recv1_cb_)
  , _comm(comm_)
  , msg_size{msg_size_}
  , br{
    }
  , bt{
      _comm, buffer_size_, remote_key_*3U+2, msg_size
    }
  , iterations_left(iteration_count_)
{
  for ( auto i = 0U; i != 2U; ++i )
  {
    br.emplace_back(_comm, buffer_size_, remote_key_*3U+1, msg_size);
  }
}

cnxn_state::cnxn_state(cnxn_state &&cs_)
  : send_ctxt(this, cs_.send_ctxt.cb_call)
  , recv0_ctxt(this, cs_.recv0_ctxt.cb_call)
  , recv1_ctxt(this, cs_.recv1_ctxt.cb_call)
  , _comm(cs_._comm)
  , msg_size(std::move(cs_.msg_size))
  , br(std::move(cs_.br))
  , bt(std::move(cs_.bt))
  , iterations_left(std::move(cs_.iterations_left))
{
}

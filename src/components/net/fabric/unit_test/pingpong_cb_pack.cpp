#include "pingpong_cb_pack.h"

cb_pack::cb_pack(
  cnxn_state &state_
  , Component::IFabric_connection &cnxn_
  , cb_ctxt::cb_t send_cb
  , cb_ctxt::cb_t recv_cb
  , std::uint64_t buffer_size_
  , std::uint64_t remote_key_base_
  , std::uint64_t msg_size_
)
  : _tx_ctxt(state_, send_cb, nullptr, cnxn_, buffer_size_, remote_key_base_*3U, msg_size_, "tx")
  , _rx0_ctxt(state_, recv_cb, &_tx_ctxt, cnxn_, buffer_size_, remote_key_base_*3U+1, msg_size_, "rx0")
  , _rx1_ctxt(state_, recv_cb, &_tx_ctxt, cnxn_, buffer_size_, remote_key_base_*3U+2, msg_size_, "rx1")
{
}

cb_pack::~cb_pack()
{
}

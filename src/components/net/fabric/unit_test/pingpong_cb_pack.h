#ifndef _TEST_PINGPONG_CB_PACK_H_
#define _TEST_PINGPONG_CB_PACK_H_

#include "delete_copy.h"
#include "pingpong_cb_ctxt.h"

namespace Component
{
  class IFabric_connection;
}

struct cb_pack
{
  cb_ctxt _tx_ctxt;
  cb_ctxt _rx0_ctxt;
  cb_ctxt _rx1_ctxt;
  explicit cb_pack(
    cnxn_state &state_
    , Component::IFabric_connection &cnxn_
    , cb_ctxt::cb_t send_cb
    , cb_ctxt::cb_t recv_cb
    , std::uint64_t buffer_size_
    , std::uint64_t remote_key_base_
    , std::uint64_t msg_size_
  );
  DELETE_COPY(cb_pack);
  ~cb_pack();
};

#endif

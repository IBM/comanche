#ifndef _TEST_PINGPONG_SERVER_CB_H_
#define _TEST_PINGPONG_SERVER_CB_H_

#include <common/types.h> /* status_t */
#include <cstddef> /* size_t */

class cb_ctxt;

namespace pingpong_server_cb
{
  void recv_cb(cb_ctxt *rx_ctxt_, ::status_t stat_, std::size_t len_);
  void send_cb(cb_ctxt *tx_ctxt_, ::status_t stat_, std::size_t len_);
}

#endif

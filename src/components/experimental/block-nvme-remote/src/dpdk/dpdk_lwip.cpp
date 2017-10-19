#include <common/logging.h>

#include "dpdk_lwip.h"
#include "lwip/init.h"

DPDK::Lwip::Lwip(DPDK::Ethernet_device& dev) :
  _ethernet_device(dev)
{
  lwip_init();
  
  IP4_ADDR(&_gw, 10,0,0,33);
  IP4_ADDR(&_ipaddr, 10,0,0,33);
  IP4_ADDR(&_netmask, 255,255,255,0);

}
    

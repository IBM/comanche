#ifndef __DPDK_LWIP_H__
#define __DPDK_LWIP_H__

#include "lwip/ip_addr.h"
#include "lwip/etharp.h"

#include "dpdk_eth.h"

namespace DPDK
{

class Lwip
{
public:
  Lwip(DPDK::Ethernet_device& dev);

private:
  DPDK::Ethernet_device& _ethernet_device;

  ip_addr_t _ipaddr;
  ip_addr_t _netmask;
  ip_addr_t _gw;
};

}

#endif // __DPDK_LWIP_H__

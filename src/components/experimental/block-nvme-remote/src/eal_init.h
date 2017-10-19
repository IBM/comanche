/*
 * (C) Copyright IBM Corporation 2017. All rights reserved.
 *
 */

/* 
 * Authors: 
 * 
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#ifndef __EAL_INIT_H__
#define __EAL_INIT_H__

#include <rte_config.h>
#include <rte_malloc.h>
#include <rte_mempool.h>
#include <string>

#if !defined(__cplusplus)
#error "eal_init.h is C++ only"
#endif

namespace DPDK {

void eal_init(size_t memory_limit_MB);

}

#endif

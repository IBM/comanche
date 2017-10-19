/*
 * (C) Copyright IBM Corporation 2017. All rights reserved.
 *
 * Specifications of message types between kernel and userspace
 */

/* 
 * Authors: 
 * 
 * Feng Li (feng.li1@ibm.com)
 *
 */

#ifndef COPAGER_MSG
#define COPAGER_MSG

#include <linux/ioctl.h>

#define DEBUG
#define FIFO_SIZE 32

#define COPAGER_IOCTL_TAG_SERVICE 3
 
#define ADDR_IDX_FAULT 0
#define ADDR_IDX_PHYS  0
#define ADDR_IDX_INVAL 1

typedef struct
{
  struct {
    volatile int * signal;
    void *         this_ptr;
    int            pid;
    unsigned long  addr[2];
  }; 
} __attribute__((packed)) service_request_t;


#endif


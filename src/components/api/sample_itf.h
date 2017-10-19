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
#ifndef __API_SAMPLE_ITF__
#define __API_SAMPLE_ITF__

#include <common/types.h>
#include <api/components.h>

namespace Component
{


/** 
 * ISample - demo interface
 */
class ISample : public Component::IBase
{
public:
  /* generate from uuidgen command line tool - do not reuse this one */
  DECLARE_INTERFACE_UUID(0xbeb21d37,0x503c,0x4643,0xafa5,0x40,0x5f,0xf6,0x4c,0x90,0xc1);

public:

  /* example method */
  virtual void say_hello() = 0;
};


class ISample_factory : public Component::IBase
{
public:
  /* generate from uuidgen command line tool or derive from above - do not reuse this one */
  DECLARE_INTERFACE_UUID(0xfac21d37,0x503c,0x4643,0xafa5,0x40,0x5f,0xf6,0x4c,0x90,0xc1);

  /* simple factory instance creation point */
  virtual ISample * open(std::string name) = 0;
};


}

#endif 

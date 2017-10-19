/*
 * (C) Copyright IBM Corporation 2017. All rights reserved.
 *
 */

/* 
 * Authors: 
 * 
 * Feng Li (feng.li1@ibm.com)
 */
#pragma once

#ifndef __CIR_BUFF__
#define __CIR_BUFF__

#include <iostream>


typedef unsigned long addr_t; 


class Ring_buffer
{
  /*
   * use fix array to implement deque,
   */
  private:
        addr_t * _buffer ; // virtaddr allocated in  pinned mem
        size_t _first ;
        size_t _last;
        size_t _cnt; // number of items in 
        size_t _size;
  public:
    Ring_buffer(addr_t *p, size_t num_elem):_buffer(p){

      _size = num_elem;
      _first = 0;
      _last = 0;
      _cnt = 0;
    }

    bool empty() const { return _cnt == 0 ; }
    bool full() const { return _cnt == _size; }

    int push_back( addr_t tmp)
    {
      if(full())
        return 0;
      else{
        _buffer[_last] = tmp ;
        _last = (_last+1)%(_size) ;
        ++_cnt ;
        return 1;
      }
    }

   int pop_front(addr_t *tmp)
    {
      if(empty())
        return 0;
      else{

        *tmp = _buffer[_first];
        _first = (_first+1)%(_size) ;// 
        --_cnt ;
        return 1;
      }
   }

   size_t length(){
     return _cnt;
   }

   void print(){
     addr_t tmp;
     while(!empty()){
       if(pop_front(&tmp))
       std::cout << tmp<< std::endl;
       else
         std::cerr<< "no";
     }
   }
    
};

#endif




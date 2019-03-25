/*
   Copyright [2017-2019] [IBM Corporation]
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/


/* 
 * Authors: 
 * 
 * Feng Li (feng.li1@ibm.com)
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#pragma once


#ifndef __PTE_H__
#define __PTE_H__

#define PAGE_SHIFT (12)
#define PAGE_SIZE (4096)

#define _PAGE_PRESENT (1<<0)
#define _PAGE_ACCESSED (1<<5)


/*
 * page table implementation is user space 
 * ported from kernel source: pgtable.h
 */



typedef unsigned long pteval_t;
typedef unsigned long pg_off_t;// offset in virtual memory area(in page)
typedef unsigned long pfn_off_t; // offset in pysical page

/*
 * this struct will be used in active/free lists
 */
struct Frame{
    Frame(pg_off_t a, pfn_off_t b):virt_off(a),phys_off(b)
    {}

    pg_off_t virt_off;
    pfn_off_t phys_off;
};
typedef struct{pteval_t pte;} pte_t;


static inline int pte_present(pte_t a){
  return (a.pte&_PAGE_PRESENT);
}

static inline int pte_young(pte_t a){
  return (a.pte&_PAGE_ACCESSED);
}



#endif



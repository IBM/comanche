/*
 * (C) Copyright IBM Corporation 2017. All rights reserved.
 *
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



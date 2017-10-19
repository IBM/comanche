/* INFO: https://github.com/lorenzo-stoakes/linux-vm-notes/blob/master/sections/page-tables.md */

#include <linux/debugfs.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/seq_file.h>
#include <linux/sched.h>
#include <linux/page-flags.h>
#include <linux/bitmap.h>
#include <linux/slab.h>
#include <linux/jiffies.h>
#include <linux/hugetlb.h>
#include <asm/pgtable.h>
#include <asm/cacheflush.h>
#include <asm/tlbflush.h>
//#include <asm/pgtable_type.h>

#ifndef CONFIG_X86_64
#error This code is written for X86-64 only.
#endif

//#define DEBUG

struct bitmap_header
{
  u32 magic;
  u32 bitmap_size;
} __attribute__((packed));

/** 
 * Build bitmap. Each bit represents a single page.  Bitmap header is 32bit which indicates 
 * size of the bitmap in bytes.
 * 
 * @param start_page 
 * @param num_pages 
 * @param out_bitmap_size 
 * 
 * @return 
 */
static void * build_bitmap(u64 start_page,
                           size_t num_pages,
                           size_t * out_bitmap_size);

/*
 * The dumper groups pagetable entries of the same type into one, and for
 * that it needs to keep some state when walking, and flush this state
 * when a "break" in the continuity is found.
 */
struct pg_state {
	int level;
	pgprot_t current_prot;
	unsigned long start_phy_address;
	unsigned long current_phy_address;
	unsigned long start_address;
	unsigned long current_address;
	const struct addr_marker *marker;
};

struct addr_marker {
	unsigned long start_address;
	const char *name;
};

/* indices for address_markers; keep sync'd w/ address_markers below */
enum address_markers_idx {
	USER_SPACE_NR = 0,
#ifdef CONFIG_X86_64
	KERNEL_SPACE_NR,
	LOW_KERNEL_NR,
	VMALLOC_START_NR,
	VMEMMAP_START_NR,
	HIGH_KERNEL_NR,
	MODULES_VADDR_NR,
	MODULES_END_NR,
#else
	KERNEL_SPACE_NR,
	VMALLOC_START_NR,
	VMALLOC_END_NR,
# ifdef CONFIG_HIGHMEM
	PKMAP_BASE_NR,
# endif
	FIXADDR_START_NR,
#endif
};



/* Multipliers for offsets within the PTEs */
#define PTE_LEVEL_MULT (PAGE_SIZE)
#define PMD_LEVEL_MULT (PTRS_PER_PTE * PTE_LEVEL_MULT)
#define PUD_LEVEL_MULT (PTRS_PER_PMD * PMD_LEVEL_MULT)
#define PGD_LEVEL_MULT (PTRS_PER_PUD * PUD_LEVEL_MULT)


extern pgd_t init_level4_pgt[];

/* references

   mm_struct -->  http://lxr.free-electrons.com/source/include/linux/mm_types.h#L396
*/
pte_t* walk_page_table(u64 addr, unsigned long * huge_pfn)
{
    pgd_t *pgd;
    pte_t *ptep;
    pud_t *pud;
    pmd_t *pmd;
    struct mm_struct *mm = current->mm;
    struct vm_area_struct * vma = find_vma(mm, addr);
#ifdef DEBUG
    printk(KERN_INFO "xms: walk page table %llx", addr);
#endif
    
    pgd = pgd_offset(mm, addr);
    if (pgd_none(*pgd) || pgd_bad(*pgd)) {
      printk(KERN_ERR "xms: pdg not found\n");
      return NULL;
    }    
    
    pud = pud_offset(pgd, addr);
    if (pud_none(*pud) || pud_bad(*pud)) {
      printk(KERN_ERR "xms: pud not found\n");
      return NULL;
    }
    /* When huge pages are enabled on x86-64 (providing for 2MiB
       pages), this is achieved by setting the _PAGE_PSE flag on PMD
       entries. The PMD entries then no longer refer to a PTE page,
       but instead the page table structure is now terminated at the
       PMD and its physical address and flags refer to the physical
       page, leaving the remaining 21 bits (i.e. 2MiB) as an offset
       into the physical page. */
    pmd = pmd_offset(pud, addr);
    if (vma->vm_flags & VM_HUGETLB) {
      *huge_pfn = (unsigned long) pmd_val(*pmd);
      printk(KERN_ERR "xms: vm_flag hugepage found (%lx)\n", pmd_val(*pmd));
      return NULL;
    }
    else {
      if (pmd_none(*pmd) || pmd_bad(*pmd)) {
        printk(KERN_ERR "xms: pmd not found\n");
        return NULL;
      }
    }

    /* if (pmd_huge(*pmd) && vma->vm_flags & VM_HUGETLB) { */
    /*   printk(KERN_ERR "xms: unhandled huge page\n"); */
    /*   return NULL; */
    /*   //      return pmd_val(*pmd); */
    /* } */
     
    if(*((u64*)pmd) & (1<<7)) {
      printk(KERN_ERR "walk_page_table: not 4K page");
      return NULL;
    }

    ptep = pte_offset_map(pmd, addr);
    if (!ptep) {
      printk(KERN_ERR "xms: ptep not found\n");
      return NULL;
    }

    /* x86 page table memory is always mapped in */
    
    //    pte = *ptep;

 /*    page = pte_page(pte); */
 /*    if (!page) */
 /*      printk(KERN_INFO "xms: page not found\n"); */

 /*    pte_unmap(pte); //not needed assuming x86 */
    
    return ptep;
}

void * collect_bits(void * addr, size_t size, size_t * out_bitmap_size)
{
  /* calculate page range */
  u64 start_pg = ((u64)addr) >> PAGE_SHIFT;
  u64 end_pg = ((u64)addr + size -1) >> PAGE_SHIFT;

#ifdef DEBUG
  printk(KERN_INFO "xms: collect_bits(%p,%ld)\n",addr,size);
#endif
  
  int num_pages = end_pg - start_pg + 1;
    return build_bitmap(start_pg, num_pages, out_bitmap_size);
}

static void set_bitmap_bit(void * bitmap, size_t position)
{
  size_t dword_pos = position / (sizeof(unsigned long)*8);
  long dword_bit_pos = position % (sizeof(unsigned long)*8);
  volatile unsigned long * mem_pos =
    ((volatile unsigned long*) bitmap) + dword_pos;
  
  __set_bit(dword_bit_pos, mem_pos);
}


static bool is_last_in_4K_page(pte_t * pte)
{
  if(((unsigned long) pte++) & 0xFFF)
    return true;
  return false;
}

static void * build_bitmap(u64 start_page,
                           size_t num_pages,
                           size_t * out_bitmap_size)
{
  u64 jiffies_begin;

  // to do check page size
  struct mm_struct * mm = current->mm;  
  struct vm_area_struct *vma;
  size_t i;
  struct bitmap_header *kmem;
  void * bitmap;
  size_t kmem_size;
  size_t rounded_up_bitmap_size;
    
  size_t actual_bitmap_size = num_pages / 8;

  jiffies_begin = jiffies;
  
  vma = find_vma(mm, (start_page << PAGE_SHIFT));

  if(vma == NULL) {
    printk(KERN_ERR "page_walk: failed to find_vma (page=%llx)\n", start_page);
    return NULL;
  }

  if(num_pages % 8) actual_bitmap_size ++;

  rounded_up_bitmap_size = actual_bitmap_size;

  while(rounded_up_bitmap_size % 4)
    rounded_up_bitmap_size++; // bitmap setting uses 32bit data 

#ifdef DEBUG
  printk(KERN_INFO "xms: bitmap size = %ld bytes\n", rounded_up_bitmap_size);
#endif
  kmem_size = rounded_up_bitmap_size + sizeof(struct bitmap_header); 
  kmem = kmalloc(kmem_size, GFP_KERNEL);
  memset(kmem,0,kmem_size);

  /* reserve space for bitmap header */
  bitmap = ((char*)kmem) + sizeof(struct bitmap_header);

#if 0 /* move cache flush to user-land to avoid needing to map memory into kernel */
  /* this code will fault on the memory address being flushed */
  /* flush cache */
  {
    u64 jiffies_begin = jiffies;
    clflush_cache_range(((void*)(start_page << PAGE_SHIFT)), PAGE_SIZE * num_pages);
    printk(KERN_INFO "flushed %ld pages OK (%u usecs)\n",
           num_pages,
           jiffies_to_usecs(jiffies - jiffies_begin));
  }
#endif
  
  {
    pte_t * pte = NULL;
    
    for(i=0;i<num_pages;i++) {

      /* todo: optimize to avoid re-walking, increment pte until next 4K boundary */
          
      u64 addr = (start_page + i) << PAGE_SHIFT;
      unsigned long huge_pfn = 0;
      if(!pte || is_last_in_4K_page(pte))
        pte = walk_page_table(addr, &huge_pfn);
      else 
        pte++;

      if(pte == NULL) {
        printk(KERN_ERR "build_bitmap: no page entry\n");
        continue;
      }

      if(pte_present(*pte)) {
#ifdef DEBUG
        if(pte_dirty(*pte)) {
          printk(KERN_INFO "page: %llx %llx dirty=%s accessed=%s pfn=%lx\n",
                 addr,
                 *((u64*)pte),
                 pte_dirty(*pte) ? "y":"n",
                 pte_young(*pte) ? "y":"n",
                 pte_pfn(*pte)
                 );
        }
#endif

        if(pte_dirty(*pte)) {
          set_bitmap_bit(bitmap, i);
        }
      }
#ifdef DEBUG
      else {
        printk(KERN_INFO "nopage: %llx", addr);
      }
#endif
    
      /* clear dirty and accessed */
      *pte = pte_clear_flags(*pte, _PAGE_DIRTY | _PAGE_ACCESSED);

      __flush_tlb_single(addr); /* TODO: this won't work for SMP ?? */

      pte_unmap(pte);
    }
  }

  kmem->magic = 0xB0B0FE00;
  kmem->bitmap_size = actual_bitmap_size;
  *out_bitmap_size = kmem_size;

#ifdef DEBUG
  printk(KERN_INFO "xms: bitmap size = %d", kmem->bitmap_size);
  printk(KERN_INFO "xms: build_bitmap %u usec",
         jiffies_to_usecs(jiffies - jiffies_begin));
#endif
  
  return kmem;
}

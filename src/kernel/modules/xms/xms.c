/**
 * @file    xms.c
 * @author  Daniel Waddington
 * @date    10 August 2017
 * @version 0.1
 * @brief   Xms support module.
 */

#include <linux/version.h>
#include <linux/init.h>             // Macros used to mark up functions e.g., __init __exit
#include <linux/module.h>           // Core header for loading LKMs into the kernel
#include <linux/kernel.h>           // Contains types, macros, functions for the kernel
#include <linux/errno.h>
#include <linux/kernel.h>
#include <linux/mm.h>
#include <linux/types.h>
#include <linux/miscdevice.h>
#include <linux/compat.h>
#include <linux/fs.h>
#include <linux/slab.h>
#include <linux/uaccess.h>

#if LINUX_VERSION_CODE < KERNEL_VERSION(4,13,0)
#include <asm/cacheflush.h>
#else
#include <asm/set_memory.h>
#endif

/* Linux 5 drops the mode atribute in access_ok */
#if LINUX_VERSION_CODE < KERNEL_VERSION(5,0,0)
#define ACCESS_OK(mode,addr,size) access_ok(mode,addr,size)
#else
#define ACCESS_OK(mode,addr,size) access_ok(addr,size)
#endif

#include <asm/pgtable.h>
#include <asm/uaccess.h>

//#define DEBUG

#include "xms.h"

MODULE_LICENSE("GPL");              ///< The license type -- this affects runtime behavior
MODULE_AUTHOR("Daniel Waddington"); ///< The author -- visible when you use modinfo
MODULE_DESCRIPTION("eXtra Memory Service module.");  ///< The description -- see modinfo
MODULE_VERSION("0.1");              ///< The version of the module

// forward decls
//
void * collect_bits(void * addr, size_t size, size_t * out_bitmap_size);
pte_t* walk_page_table(u64 addr, unsigned long * huge_pfn);

static int fop_mmap(struct file *file, struct vm_area_struct *vma);
static int set_memory_type(void* addr, size_t size, memory_t type);

inline static bool check_aligned(void* p, unsigned alignment)
{
  return (!(((unsigned long)p) & (alignment - 1UL)));
}

/* static char *name = "world"; ///< An example LKM argument --
   default value is "world" */
/* module_param(name, charp, S_IRUGO); ///< Param desc. charp = char
   ptr, S_IRUGO can be read/not changed */
/* MODULE_PARM_DESC(name, "The name to display in /var/log/kern.log");
///< parameter description */

static long xms_dev_ioctl(struct file *filp,
                          unsigned int ioctl,
                          unsigned long arg)
{
  long r = -EINVAL;
  void * bitmap;
  size_t bitmap_size = 0;

#ifdef DEBUG
  printk(KERN_INFO "xms: dev_ioctl (%d) (arg=%lx)\n", ioctl, arg);
#endif

  if(ioctl == IOCTL_CMD_SETMEMORY) {
    int rc;
    IOCTL_SETMEMORY_param params;
    rc = copy_from_user(&params,
                        ((IOCTL_SETMEMORY_param *) arg),
                        sizeof(IOCTL_SETMEMORY_param));
    
    if((rc > 0) || (!ACCESS_OK(VERIFY_READ, params.vaddr, params.size))) {
      printk(KERN_ERR "xms: dev_ioctl passed invalid param\n");
      return r;
    }

    if(!check_aligned(params.vaddr, PAGE_SIZE) || (params.size % PAGE_SIZE)) {
      printk(KERN_ERR "xms: ioctl set_memory requires page-aligned pointer (%p)\n", params.vaddr);
      return r;
    }

    return set_memory_type(params.vaddr, params.size, params.type);
  }
  else if(ioctl == IOCTL_CMD_GETPHYS) {
    IOCTL_GETPHYS_param params;
    int rc;
    
    rc = copy_from_user(&params,
                        ((IOCTL_GETPHYS_param *) arg),
                        sizeof(IOCTL_GETPHYS_param));

    if(rc > 0) {

      printk(KERN_ERR "xms: dev_ioctl passed invalid out_data param\n");
      return r;
    }
    
    if(!ACCESS_OK(VERIFY_READ, params.vaddr, sizeof(params.vaddr))) {
      printk(KERN_ERR "xms: dev_ioctl passed invalid in_data param");
      return r;
    }

    {
      int rc;
      unsigned long huge_pfn = 0;
      pte_t* pte = walk_page_table(params.vaddr, &huge_pfn);
      
      if(pte == NULL && huge_pfn > 0) {
#ifdef DEBUG
        printk(KERN_INFO "xms: huge %p->%lx",
               (void*) params.vaddr, huge_pfn);
#endif
        params.out_paddr = huge_pfn;
      }
      else {
        if(pte == NULL) {
          printk(KERN_INFO "xms: result of walk (huge page) is null");
          return r;
        }
        params.out_paddr = pte_pfn(*pte) << PAGE_SHIFT;
      }

#ifdef DEBUG
      printk(KERN_INFO "xms: page %p->%lx",
             (void*) params.vaddr, params.out_paddr);
#endif

      rc = copy_to_user(((IOCTL_GETPHYS_param *) arg),
                        &params,
                        sizeof(IOCTL_GETPHYS_param));
      if(rc > 0) {
        printk(KERN_ERR "xms: copy_to_user failed");
      }
      return 0;
    }
  }
  else if(ioctl == IOCTL_CMD_GETBITMAP) {
    int rc;
    IOCTL_GETBITMAP_param params;
    rc = copy_from_user(&params,
                        ((IOCTL_GETBITMAP_param *) arg),
                        sizeof(IOCTL_GETBITMAP_param));
    
    if((rc > 0) || (!ACCESS_OK(VERIFY_WRITE, params.out_data, params.out_size))) {
      printk(KERN_ERR "xms: dev_ioctl passed invalid out_data param");
      return r;
    }
    
    if(!ACCESS_OK(VERIFY_READ, params.ptr, params.size)) {
      printk(KERN_ERR "xms: dev_ioctl passed invalid in_data param");
      return r;
    }

    bitmap = collect_bits(params.ptr, params.size, &bitmap_size);

    if(bitmap == NULL)
      return r;
    
    if(bitmap_size > params.out_size) {
      printk(KERN_ERR "xms: dev_ioctl out_data size insufficient (bitmap=%ld,out=%ld)",
             bitmap_size, params.out_size);
      return r;
    }

    rc = copy_to_user(params.out_data, bitmap, bitmap_size);
    BUG_ON(rc > 0);
    kfree(bitmap);
    
    return 0;
  }
              
  return r;
}

static addr_t get_physical(void * vaddr)
{
  unsigned long huge_pfn = 0;
  pte_t* pte = walk_page_table((unsigned long) vaddr, &huge_pfn);
      
  if(pte == NULL && huge_pfn > 0) {
#ifdef DEBUG
    printk(KERN_INFO "xms: huge %p->%lx", (void*) vaddr, huge_pfn);
#endif
    return huge_pfn;
  }
  else {
    if(pte == NULL) return -EINVAL;
    return (pte_pfn(*pte) << PAGE_SHIFT);
  }
  return -1;
}
  

static int set_memory_type(void* vaddr, size_t size, memory_t type)
{
  addr_t paddr = get_physical(vaddr);
  unsigned num_pages = size / PAGE_SIZE;
  
  printk(KERN_INFO "set_memory_type: %p %lu %u (%u pages)\n", vaddr, size, type, num_pages);
  printk(KERN_INFO "set_memory_type: phys=0x%lx\n", paddr);

  if(type == MEMORY_UC) {
    /* int ret; */
    /* ret = reserve_memtype(paddr, paddr + num_pages * PAGE_SIZE, */
    /*                       _PAGE_CACHE_MODE_UC_MINUS, NULL); */
    //    ret = reserve_memtype(
    //    rc = set_memory_uc(paddr, num_pages);
  }
  return 0;
}

static int xms_dev_release(struct inode *inode, struct file *file)
{
  return 0;
}



static const struct file_operations xms_chardev_ops = {
  .owner          = THIS_MODULE,
  .unlocked_ioctl = xms_dev_ioctl,
#ifdef CONFIG_COMPAT
  .compat_ioctl   = xms_dev_ioctl,
#endif
  .llseek         = noop_llseek,
  .release        = xms_dev_release,
  .mmap           = fop_mmap,
};

static struct miscdevice xms_dev = {
  MISC_DYNAMIC_MINOR,
  // XMS_MINOR,
  "xms",
  &xms_chardev_ops,
};

static void vm_open(struct vm_area_struct *vma)
{
}

static void vm_close(struct vm_area_struct *vma)
{
}

#if LINUX_VERSION_CODE < KERNEL_VERSION(4,13,0)
static int vm_fault(struct vm_area_struct *area, 
                    struct vm_fault *fdata)
#else
  static int vm_fault(struct vm_fault *vmf)
#endif
{
  return VM_FAULT_SIGBUS;
}

static struct vm_operations_struct mmap_fops = {
  .open   = vm_open,
  .close  = vm_close,
  .fault  = vm_fault
};

/** 
 * Allows mmap calls to map a virtual region to a specific
 * physical address
 * 
 */
static int fop_mmap(struct file *file, struct vm_area_struct *vma)
{
  unsigned long type;
  addr_t phys;
  
  if(vma->vm_end < vma->vm_start)
    return -EINVAL;

  /* TODO: PRIV check!!!! */

#ifdef DEBUG
  printk(KERN_DEBUG "fop_mmap: flags=%lx offset=%lx\n", vma->vm_flags, vma->vm_pgoff);
#endif
  type = (vma->vm_pgoff >> 48);
  phys = (vma->vm_pgoff & 0xffffffffffffULL);

#ifdef DEBUG
  printk(KERN_DEBUG "fop_mmap: type=%lx phys=%lx\n", type, phys);
#endif
  //  unsigned long offset = vma->vm_pgoff * PAGE_SIZE;

  if(type == 1) {
    vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot); /* uncached */
    printk(KERN_DEBUG "fop_mmap: set non-cached\n");
  }
  else if(type == 2) {
    vma->vm_page_prot = pgprot_writecombine(vma->vm_page_prot); /* uncached */
    printk(KERN_DEBUG "fop_mmap: set write-combined\n");
  }
  /* PDBG("file->private_data = %p",file->private_data); */

  //vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot); /* disable cache */
  //vma->vm_page_prot = pgprot_writecombine(vma->vm_page_prot); /* cache write combine */

  /* check this area was allocated by the parasitic kernel and
     also check that it is owned by the current task
  */
  /* { */
  /*   addr_t phys = vma->vm_pgoff << PAGE_SHIFT; */
  /*   struct pk_dma_area * area = get_owned_dma_area(phys); */
  /*   if(!area) { */
  /*     printk(KERN_ERR "DMA area (%p) is not owned by caller nor is it shared",(void *) phys); */
  /*     return -EPERM; */
  /*   } */
  /* } */

  if (remap_pfn_range(vma, 
                      vma->vm_start,
                      phys, //vma->vm_pgoff, // passed through physical address 
                      vma->vm_end - vma->vm_start,
                      vma->vm_page_prot)) {
    printk(KERN_ERR "remap_pfn_range failed.");
    return -EAGAIN;
  }

#ifdef DEBUG
  printk(KERN_INFO "xms: mmap virt=%lx pgoff=%lx\n", vma->vm_start, vma->vm_pgoff);
#endif

  vma->vm_ops = &mmap_fops;
  vma->vm_flags |= VM_IO | VM_PFNMAP | VM_DONTEXPAND | VM_DONTDUMP;
  
  return 0;
}



static int __init xms_init(void) {
  int r;

  xms_dev.mode = S_IRUGO | S_IWUGO; // set permission for /dev/xms
  
  r = misc_register(&xms_dev);
  if (r) {
    printk(KERN_ERR "xms: misc device register failed\n");
  }
  printk(KERN_INFO "xms: loaded\n");
  
  return 0;
}

static void __exit xms_exit(void){
  printk(KERN_INFO "xms: unloaded\n");
  misc_deregister(&xms_dev);
}

/** @brief A module must use the module_init() module_exit() macros from linux/init.h, which
 *  identify the initialization function at insertion time and the cleanup function (as
 *  listed above)
 */
module_init(xms_init);
module_exit(xms_exit);

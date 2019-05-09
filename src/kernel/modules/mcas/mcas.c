/**
 * @file    mcas.c
 * @author  Daniel Waddington (daniel.waddington@acm.org)
 * @date    9 May 2019
 * @version 0.1
 * @brief   MCAS support module.
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

#include <asm/pgtable.h>
#include <asm/uaccess.h>

#define DEBUG

#include "mcas.h"

MODULE_LICENSE("GPL");              ///< The license type -- this affects runtime behavior
MODULE_AUTHOR("Daniel Waddington"); ///< The author -- visible when you use modinfo
MODULE_DESCRIPTION("MCAS support module.");  ///< The description -- see modinfo
MODULE_VERSION("0.1");              ///< The version of the module


static int fop_mmap(struct file *file, struct vm_area_struct *vma);

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

static long mcas_dev_ioctl(struct file *filp,
                          unsigned int ioctl,
                          unsigned long arg)
{
  long r = -EINVAL;

#ifdef DEBUG
  printk(KERN_INFO "mcas: dev_ioctl (%d) (arg=%lx)\n", ioctl, arg);
#endif
             
  return r;
}

static int mcas_dev_release(struct inode *inode, struct file *file)
{
  return 0;
}



static const struct file_operations mcas_chardev_ops = {
  .owner          = THIS_MODULE,
  .unlocked_ioctl = mcas_dev_ioctl,
#ifdef CONFIG_COMPAT
  .compat_ioctl   = mcas_dev_ioctl,
#endif
  .llseek         = noop_llseek,
  .release        = mcas_dev_release,
  .mmap           = fop_mmap,
};

static struct miscdevice mcas_dev = {
  MCAS_MINOR,
  "mcase",
  &mcas_chardev_ops,
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

/*   type = (vma->vm_pgoff >> 48); */
/*   phys = (vma->vm_pgoff & 0xffffffffffffULL); */

/* #ifdef DEBUG */
/*   printk(KERN_DEBUG "fop_mmap: type=%lx phys=%lx\n", type, phys); */
/* #endif */
/*   //  unsigned long offset = vma->vm_pgoff * PAGE_SIZE; */

/*   if(type == 1) { */
/*     vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot); /\* uncached *\/ */
/*     printk(KERN_DEBUG "fop_mmap: set non-cached\n"); */
/*   } */
/*   else if(type == 2) { */
/*     vma->vm_page_prot = pgprot_writecombine(vma->vm_page_prot); /\* uncached *\/ */
/*     printk(KERN_DEBUG "fop_mmap: set write-combined\n"); */
/*   } */
/*   /\* PDBG("file->private_data = %p",file->private_data); *\/ */

/*   //vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot); /\* disable cache *\/ */
/*   //vma->vm_page_prot = pgprot_writecombine(vma->vm_page_prot); /\* cache write combine *\/ */

/*   /\* check this area was allocated by the parasitic kernel and */
/*      also check that it is owned by the current task */
/*   *\/ */
/*   /\* { *\/ */
/*   /\*   addr_t phys = vma->vm_pgoff << PAGE_SHIFT; *\/ */
/*   /\*   struct pk_dma_area * area = get_owned_dma_area(phys); *\/ */
/*   /\*   if(!area) { *\/ */
/*   /\*     printk(KERN_ERR "DMA area (%p) is not owned by caller nor is it shared",(void *) phys); *\/ */
/*   /\*     return -EPERM; *\/ */
/*   /\*   } *\/ */
/*   /\* } *\/ */

/*   if (remap_pfn_range(vma,  */
/*                       vma->vm_start, */
/*                       phys, //vma->vm_pgoff, // passed through physical address  */
/*                       vma->vm_end - vma->vm_start, */
/*                       vma->vm_page_prot)) { */
/*     printk(KERN_ERR "remap_pfn_range failed."); */
/*     return -EAGAIN; */
/*   } */

/* #ifdef DEBUG */
/*   printk(KERN_INFO "mcas: mmap virt=%lx pgoff=%lx\n", vma->vm_start, vma->vm_pgoff); */
/* #endif */

/*   vma->vm_ops = &mmap_fops; */
/*   vma->vm_flags |= VM_IO | VM_PFNMAP | VM_DONTEXPAND | VM_DONTDUMP; */
  
  return 0;
}



static int __init mcas_init(void) {
  int r;

  mcas_dev.mode = S_IRUGO | S_IWUGO; // set permission for /dev/mcas
  
  r = misc_register(&mcas_dev);
  if (r) {
    printk(KERN_ERR "mcas: misc device register failed\n");
  }
  printk(KERN_INFO "mcas: loaded\n");
  
  return 0;
}

static void __exit mcas_exit(void){
  printk(KERN_INFO "mcas: unloaded\n");
  misc_deregister(&mcas_dev);
}

/** @brief A module must use the module_init() module_exit() macros from linux/init.h, which
 *  identify the initialization function at insertion time and the cleanup function (as
 *  listed above)
 */
module_init(mcas_init);
module_exit(mcas_exit);

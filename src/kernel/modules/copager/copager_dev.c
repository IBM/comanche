/*
 * (C) Copyright IBM Corporation 2017. All rights reserved.
 *
 */

/* 
 * Authors: 
 * 
 * Daniel Waddington (daniel.waddington@ibm.com)
 * Feng Li (feng.li1@ibm.com)
 *
 */
#include <linux/module.h>
#include <linux/init.h>
#include <linux/poll.h>
#include <linux/device.h>
#include <linux/slab.h>
#include <linux/mm.h>
#include <linux/idr.h>
#include <linux/kfifo.h>
#include <linux/string.h>
#include <linux/kobject.h>
#include <linux/cdev.h>
#include <linux/miscdevice.h>
#include <linux/wait.h>
#include <linux/poll.h>

#include "copager_msg.h"

extern int copager_fops_mmap(struct file* file, struct vm_area_struct *vma);

DECLARE_KFIFO(srq, service_request_t*, FIFO_SIZE);
DECLARE_WAIT_QUEUE_HEAD(wq);


static int copager_open(struct inode *inode, struct file * filep)
{
	return 0;
}

static int copager_release(struct inode *inode, struct file* filep)
{
  return 0;
}


long copager_ioctl(struct file *f, unsigned int cmd, unsigned long arg)
{
  service_request_t* req;
  service_request_t sr;
  int rc;
  
  switch (cmd)
    {
    case COPAGER_IOCTL_TAG_SERVICE:
      if(copy_from_user(&sr, (void*)arg, sizeof(service_request_t))) {
          printk(KERN_NOTICE "[ioctl] copy_from_user error");
          return -EACCES;
      }
#ifdef DEBUG
      printk(KERN_INFO "[copager]: down sr args:%lx %lx %d\n",
             sr.addr[0], sr.addr[1], sr.pid);
#endif

      /* process down message */
      if(sr.signal != NULL) {
        service_request_t * pfsr = sr.this_ptr;
        BUG_ON(pfsr==NULL);
        pfsr->addr[ADDR_IDX_PHYS] = sr.addr[ADDR_IDX_PHYS];
        pfsr->addr[ADDR_IDX_INVAL] = sr.addr[ADDR_IDX_INVAL];
        wmb();
        *sr.signal = 1;
        wmb();
#ifdef DEBUG
        printk(KERN_INFO "[copager]: signaled on %p\n",sr.signal);
#endif
      }
      
      /* if fifo is empty, sleelp and wait for service request */
      if(kfifo_is_empty(&srq)) {
        rc = wait_event_interruptible_timeout(wq,
                                              kfifo_is_empty(&srq) == false,
                                              HZ*5);
        
        if(rc == -512) {
          return -EINTR;
        }
        if(kfifo_is_empty(&srq)) {
          printk(KERN_INFO "[copager]: ioctl thread timed out (%d)\n",rc);
          return -1;
        }
      }

      kfifo_out(&srq, &req, 1);

#ifdef DEBUG
      printk(KERN_INFO "[copager]: (from kfifo) up sr args:%lx %p %d\n",
             req->addr[0], req->signal,req->pid);
      printk(KERN_INFO "[copager]: sending up service request\n");
#endif
      
      /* process request */
      if(copy_to_user((void*)arg, req, sizeof(service_request_t))) {
          printk(KERN_NOTICE "[copager]: copy_to_user error");
          return -EACCES;
      }
      return 0; // timeout
        
    default:
      printk(KERN_INFO "unknown ioctl (%d)\n", cmd);
      return -EINVAL;
    }
 
  return 0;
}



// mmap is a field in file_operation, see understand linux kernel P.596
// use def_blk_fops
// how to hook to blk_device?
static const struct file_operations copager_misc_fops = {
	.owner =	        THIS_MODULE,
	.open =		        copager_open,
	.release =	      copager_release,
  .mmap =           copager_fops_mmap,
  .unlocked_ioctl = copager_ioctl,
};

static struct miscdevice copager_miscdev = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "copager",
	.fops = &copager_misc_fops,
  .mode = S_IRUGO | S_IWUGO,
};

/** 
 * Register the module as a misc device.  Memory services etc. are provided
 * through a top-level ioctl service.
 */
static int __init copager_init(void)
{

  if (misc_register(&copager_miscdev))
    panic("misc_register failed unexpectedly.");

  INIT_KFIFO(srq);
  
	pr_info("copager: module loaded\n");

	return 0;
}

static void __exit copager_exit(void)
{
  misc_deregister(&copager_miscdev);

	pr_info("copager: module unloaded\n");
}

module_init(copager_init);
module_exit(copager_exit);

MODULE_AUTHOR("Feng Li <feng.li1@ibm.com>");
MODULE_LICENSE("GPL");

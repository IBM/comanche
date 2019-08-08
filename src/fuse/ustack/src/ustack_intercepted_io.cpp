#include <common/utils.h>
#include <common/str_utils.h>
#include <csignal>
#include <future>
#include "ustack.h"
#include "kv_ustack_info.h"
#include "protocol_generated.h"

  /*
   * Do the write through the kvfs daemon
   *
   * @param client_id  client who issue this io, we need this to find the right iomem
   * @param fuse_fh fuse daemon file handler
   * @param offset offset of iobuffer inside the iomem
   * @param io_sz  I/O size
   * @param file_off file offset
   *
   * For posix calls , each open has its own file position maintained, But we have to do this manually here
   */
  status_t Ustack::do_kv_write(pid_t client_id, uint64_t fuse_fh, size_t offset, size_t io_sz, size_t file_off ){
    PDBG("[%s]: fuse_fh=%lu, offset=%lu, io_sz=%lu, file_off=%lu", __func__, fuse_fh, offset, io_sz, file_off);

    //get the virtual address and issue the io
    void *buf; //the mapped io mem

    // TODO: need to control offset 
    auto iomem_list = _iomem_map[client_id] ;
    if(iomem_list.empty()){
      PERR("iomem for pid %d is empty", client_id);
      return -1;
    }
    auto iomem = iomem_list[0];

    buf = iomem->offset_to_virt(offset);
    //TODO: check file 
    if(buf == 0){
      PERR("mapped virtual address failed");
      return -1;
    }

    if(io_sz >0){
      _kv_ustack_info->write(fuse_fh, buf, io_sz, file_off);
    }
    else PWRN("io size = 0");
    
    return S_OK;
  }

  status_t Ustack::do_kv_read(pid_t client_id, uint64_t fuse_fh, size_t offset, size_t io_sz, size_t file_off){
    PDBG("[%s]: fuse_fh=%lu, offset=%lu, io_sz=%lu, file_off=%lu", __func__, fuse_fh, offset, io_sz, file_off);
    void *buf; //the mapped io mem

    auto iomem_list = _iomem_map[client_id] ;
    if(iomem_list.empty()){
      PERR("iomem for pid %d is empty", client_id);
      return -1;
    }
    auto iomem = iomem_list[0];

    buf = iomem->offset_to_virt(offset);
    //TODO: check file 
    if(buf == 0){
      PERR("mapped virtual address failed");
      return -1;
    }
    PDBG("mapped to virtual address %p", buf);

    size_t file_size = _kv_ustack_info->get_item_size(fuse_fh);
    if(io_sz > file_size){
      PERR("write size larger than file size");
      return -1;
    }

    if(io_sz >0){
      _kv_ustack_info->read(fuse_fh, buf, io_sz, file_off);
    }
      else PWRN("io size = 0");
    return S_OK;
  }


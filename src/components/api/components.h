#ifndef __COMPONENTS_H__
#define __COMPONENTS_H__

#include <component/base.h>
/* Note: these uuid decls are so we don't have to have access to the component source code */

#include "itf_ref.h"
#include "types.h"

namespace Component
{

/*< sample-component - copy from implementation file */
DECLARE_STATIC_COMPONENT_UUID(sample_component_factory, 0xfac64581,0x9e1b,0x4811,0xbdb2,0x19,0x57,0xa0,0xa6,0x84,0x57);
DECLARE_STATIC_COMPONENT_UUID(sample_component, 0x59564581,0x9e1b,0x4811,0xbdb2,0x19,0x57,0xa0,0xa6,0x84,0x57);

/*< block-nvme component (based on SPDK)  */
DECLARE_STATIC_COMPONENT_UUID(block_nvme_factory,0xFAC2215b,0xff57,0x4efd,0x9e4e,0xca,0xe8,0x9b,0x70,0x73,0x2a);
DECLARE_STATIC_COMPONENT_UUID(block_nvme,0x24518582,0xd731,0x4406,0x9eb1,0x58,0x70,0x26,0x40,0x8e,0x23);

/*< block-unvme component (based on Micron's UNVMe) */
DECLARE_STATIC_COMPONENT_UUID(block_unvme_factory,0xface284f,0x9c09,0x4e7d,0x8308,0x3a,0xa4,0xb0,0xc6,0x7d,0xfd);
DECLARE_STATIC_COMPONENT_UUID(block_unvme,0x1efe284f,0x9c09,0x4e7d,0x8308,0x3a,0xa4,0xb0,0xc6,0x7d,0xfd);

/*< block-posix component (based on POSIX) */
DECLARE_STATIC_COMPONENT_UUID(block_posix_factory,0xfacf047a,0x329c,0x4452,0xa75e,0x5a,0x1c,0xa2,0xe8,0xe4,0xc4);
DECLARE_STATIC_COMPONENT_UUID(block_posix,0x1a2f047a,0x329c,0x4452,0xa75e,0x5a,0x1c,0xa2,0xe8,0xe4,0xc4);

/*< fs-minix */
DECLARE_STATIC_COMPONENT_UUID(fs_minix,0xf53fd819,0xe157,0x4e69,0x9157,0xe8,0x49,0x51,0x77,0xa3,0x25);

/*< part-gpt */
DECLARE_STATIC_COMPONENT_UUID(part_gpt_factory,0xfac2ccc2,0x28bb,0x41be,0x8b43,0x07,0xaa,0x90,0x76,0xf8,0x84);
DECLARE_STATIC_COMPONENT_UUID(part_gpt,0xda92ccc2,0x28bb,0x41be,0x8b43,0x07,0xaa,0x90,0x76,0xf8,0x84);

/*< part-region */
DECLARE_STATIC_COMPONENT_UUID(part_region_factory,0xfac2ccc2,0x28bb,0x41be,0x8b43,0x07,0xaa,0x90,0x76,0xf8,0x84);
DECLARE_STATIC_COMPONENT_UUID(part_region,0x7d55365a,0xeccc,0x4e13,0x9281,0xaa,0xc9,0xfc,0x5b,0xc5,0xa9);

/*< blob */
DECLARE_STATIC_COMPONENT_UUID(blob_factory,0xfacd40ac,0x5a61,0x4489,0xae09,0xcf,0x7e,0x29,0x8b,0xb9,0x90);
DECLARE_STATIC_COMPONENT_UUID(blob,0x320d40ac,0x5a61,0x4489,0xae09,0xcf,0x7e,0x29,0x8b,0xb9,0x90);

/*< pmem-paged */
DECLARE_STATIC_COMPONENT_UUID(pmem_paged_factory, 0xfac39351,0xa08b,0x4d43,0xb4cc,0xf4,0x88,0x92,0xac,0x4f,0x77);
DECLARE_STATIC_COMPONENT_UUID(pmem_paged,0x8a239351,0xa08b,0x4d43,0xb4cc,0xf4,0x88,0x92,0xac,0x4f,0x77);

/*< pager-simple */
DECLARE_STATIC_COMPONENT_UUID(pager_simple_factory,0xfacfd819,0xe157,0x4e69,0x9157,0xe8,0x49,0x51,0x77,0x07,0x17);
DECLARE_STATIC_COMPONENT_UUID(pager_simple,0xf53fd819,0xe157,0x4e69,0x9157,0xe8,0x49,0x51,0x77,0x07,0x17);

/*< pmem-fixed */
DECLARE_STATIC_COMPONENT_UUID(pmem_fixed_factory, 0xfac62849,0xb0ea,0x41a3,0xaed8,0x8f,0x9d,0xfb,0x4a,0xf4,0xaa);
DECLARE_STATIC_COMPONENT_UUID(pmem_fixed, 0x21562849,0xb0ea,0x41a3,0xaed8,0x8f,0x9d,0xfb,0x4a,0xf4,0xaa);

/*< block-alloc */
DECLARE_STATIC_COMPONENT_UUID(block_allocator_factory,0xfac3a368,0xd93b,0x488b,0x95cf,0x8c,0x77,0x3c,0xc5,0xaa,0xf3);
DECLARE_STATIC_COMPONENT_UUID(block_allocator,0x7f23a368,0xd93b,0x488b,0x95cf,0x8c,0x77,0x3c,0xc5,0xaa,0xf3);

/*< store-append */
DECLARE_STATIC_COMPONENT_UUID(store_append_factory, 0xfacf7650,0xe7b8,0x4747,0xb6ea,0x46,0xe1,0x09,0xf5,0x99,0x97);
DECLARE_STATIC_COMPONENT_UUID(store_append, 0x679f7650,0xe7b8,0x4747,0xb6ea,0x46,0xe1,0x09,0xf5,0x99,0x97);

/*< store-log */
DECLARE_STATIC_COMPONENT_UUID(store_log_factory, 0xfac2ad2a,0xaf0d,0x4834,0xbfc0,0x93,0x13,0x1a,0x3e,0xd7,0xb0);
DECLARE_STATIC_COMPONENT_UUID(store_log, 0xef92ad2a,0xaf0d,0x4834,0xbfc0,0x93,0x13,0x1a,0x3e,0xd7,0xb0);

/*< block-raid */
DECLARE_STATIC_COMPONENT_UUID(block_raid, 0x31197cbd,0xe4c8,0x471b,0x9296,0xb9,0x45,0xbf,0x6b,0xca,0xdb);

/*< rdma */
DECLARE_STATIC_COMPONENT_UUID(net_rdma,0x7b2b8cb7,0x5747,0x40e9,0x915d,0x69,0x43,0xa8,0xc5,0x9a,0x60);
DECLARE_STATIC_COMPONENT_UUID(net_rdma_factory,0xfacb8cb7,0x5747,0x40e9,0x915d,0x69,0x43,0xa8,0xc5,0x9a,0x60);

/*< fabric */
DECLARE_STATIC_COMPONENT_UUID(net_fabric,0x8b93a5ae,0xcf34,0x4aff,0x8321,0x19,0x08,0x21,0xa9,0x9f,0xd3);
DECLARE_STATIC_COMPONENT_UUID(net_fabric_factory,0xfac3a5ae,0xcf34,0x4aff,0x8321,0x19,0x08,0x21,0xa9,0x9f,0xd3);

/*< metadata-fixobd */
DECLARE_STATIC_COMPONENT_UUID(metadata_fixobd, 0xb2220906,0x1eec,0x48ec,0xa643,0x29,0xeb,0x6c,0x06,0x70,0xd2);
DECLARE_STATIC_COMPONENT_UUID(metadata_fixobd_factory, 0xfac20906,0x1eec,0x48ec,0xa643,0x29,0xeb,0x6c,0x06,0x70,0xd2);

/*< pmem store */
DECLARE_STATIC_COMPONENT_UUID(pmstore, 0x59564581,0x9e1b,0x4811,0xbdb2,0x19,0x57,0xa0,0xa6,0x84,0x57);
DECLARE_STATIC_COMPONENT_UUID(pmstore_factory, 0xfac64581,0x9e1b,0x4811,0xbdb2,0x19,0x57,0xa0,0xa6,0x84,0x57);

/*< pmem store */
DECLARE_STATIC_COMPONENT_UUID(nvmestore, 0x59564581,0x1993,0x4811,0xbdb2,0x19,0x57,0xa0,0xa6,0x84,0x57);
DECLARE_STATIC_COMPONENT_UUID(nvmestore_factory, 0xfac64581,0x1993,0x4811,0xbdb2,0x19,0x57,0xa0,0xa6,0x84,0x57);


/*< rocksdb */
DECLARE_STATIC_COMPONENT_UUID(rocksdb,0x0a1781e5,0x2db9,0x4876,0xb492,0xe2,0x5b,0xfe,0x17,0x3a,0xac);
DECLARE_STATIC_COMPONENT_UUID(rocksdb_factory, 0xfac781e5,0x2db9,0x4876,0xb492,0xe2,0x5b,0xfe,0x17,0x3a,0xac);

/*< filestore */
DECLARE_STATIC_COMPONENT_UUID(filestore, 0x8a120985,0xe253,0x404d,0x94d7,0x77,0x92,0x75,0x22,0xa9,0x20);
DECLARE_STATIC_COMPONENT_UUID(filestore_factory, 0xfac20985,0xe253,0x404d,0x94d7,0x77,0x92,0x75,0x22,0xa9,0x20);

/*< zyre */
DECLARE_STATIC_COMPONENT_UUID(cluster_zyre,0x5d19463b,0xa29d,0x4bc1,0x989c,0xbe,0x74,0x0a,0xc2,0x79,0x10);
DECLARE_STATIC_COMPONENT_UUID(cluster_zyre_factory,0xfac9463b,0xa29d,0x4bc1,0x989c,0xbe,0x74,0x0a,0xc2,0x79,0x10);

}


#endif // __COMPONENTS_H__


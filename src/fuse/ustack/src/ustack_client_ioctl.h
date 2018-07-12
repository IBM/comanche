#include <sys/types.h>
#include <sys/uio.h>
#include <sys/ioctl.h>

enum {
	USTACK_GET_FUSE_FH	= _IOR('E', 0, uint64_t), //the fuse_file_info->fh in fuse daemon
};

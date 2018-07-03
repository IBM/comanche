/*
 * (C) Copyright IBM Corporation 2018. All rights reserved.
 *
 */

/* 
 * Authors: 
 * 
 * Feng Li (fengggli@yahoo.com)
 */

/*
 * Simple fuse integration of ikv interface.
 *
 * Here "simple" means it uses default fuse interfaces for all file operations, 
 * which bounce between kernel/user space.
 * This can be improved greatly by intercept some of the intensive operations and handle them completel in userspace (as it's done in fuse/ustack)
 *
 * This is modified from the  hello example in the libfuse source tree.
 */

#define FUSE_USE_VERSION 26

#include <fuse.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>

static const char *kvfs_simple_str = "Hello World!\n";
static const char *kvfs_simple_path = "/hello";

static int kvfs_simple_getattr(const char *path, struct stat *stbuf)
{
	int res = 0;

	memset(stbuf, 0, sizeof(struct stat));
	if (strcmp(path, "/") == 0) {
		stbuf->st_mode = S_IFDIR | 0755;
		stbuf->st_nlink = 2;
	} else if (strcmp(path, kvfs_simple_path) == 0) {
		stbuf->st_mode = S_IFREG | 0444;
		stbuf->st_nlink = 1;
		stbuf->st_size = strlen(kvfs_simple_str);
	} else
		res = -ENOENT;

	return res;
}

static int kvfs_simple_readdir(const char *path, void *buf, fuse_fill_dir_t filler,
			 off_t offset, struct fuse_file_info *fi)
{
	(void) offset;
	(void) fi;

	if (strcmp(path, "/") != 0)
		return -ENOENT;

	filler(buf, ".", NULL, 0);
	filler(buf, "..", NULL, 0);
	filler(buf, kvfs_simple_path + 1, NULL, 0);

	return 0;
}

static int kvfs_simple_open(const char *path, struct fuse_file_info *fi)
{
	if (strcmp(path, kvfs_simple_path) != 0)
		return -ENOENT;

	if ((fi->flags & 3) != O_RDONLY)
		return -EACCES;

	return 0;
}

static int kvfs_simple_read(const char *path, char *buf, size_t size, off_t offset,
		      struct fuse_file_info *fi)
{
	size_t len;
	(void) fi;
	if(strcmp(path, kvfs_simple_path) != 0)
		return -ENOENT;

	len = strlen(kvfs_simple_str);
	if (offset < len) {
		if (offset + size > len)
			size = len - offset;
		memcpy(buf, kvfs_simple_str + offset, size);
	} else
		size = 0;

	return size;
}



int main(int argc, char *argv[])
{
  static struct fuse_operations oper;
  memset(&oper, 0, sizeof(struct fuse_operations));
	oper.getattr	= kvfs_simple_getattr;
	oper.readdir	= kvfs_simple_readdir;
	oper.open		= kvfs_simple_open;
	oper.read		= kvfs_simple_read;

	return fuse_main(argc, argv, &oper, NULL);
}

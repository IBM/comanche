#ifndef __SQLITE_EXT_MD_H__
#define __SQLITE_EXT_MD_H__

#ifdef __cplusplus
extern "C"
{
#endif

  void *  mddb_create_instance(const char * block_device_pci, int core);
  void    mddb_free_instance(void *);
  void    mddb_foo(void *);
  char *  mddb_get_schema(void *);
  
#ifdef __cplusplus
}
#endif

#endif // __SQLITE_EXT_MD_H__

#ifndef __SQLITE_EXT_MD_H__
#define __SQLITE_EXT_MD_H__

#ifdef __cplusplus
extern "C"
{
#endif

  void *  mddb_create_instance(const char * block_device_pci,
                               unsigned partition,
                               const char * owner,
                               int core);
  void    mddb_free_instance(void *);
  char *  mddb_get_schema(void *);
  void    mddb_check_canary(void *);
  
#ifdef __cplusplus
}
#endif

#endif // __SQLITE_EXT_MD_H__

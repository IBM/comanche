#include <mutex>
#include <libpmemobj.h>
#include <libpmempool.h>
#include <libpmemobj/base.h>
#include <common/exceptions.h>
#include "dax_map.h"

#define REGION_NAME "dawn-dax-pool"

static int check_pool(const char * path)
{
  PMEMpoolcheck *ppc;
  struct pmempool_check_status * status;

  struct pmempool_check_args args;
  args.path = path;
  args.backup_path = NULL;
  args.pool_type = PMEMPOOL_POOL_TYPE_DETECT;
  args.flags =
    PMEMPOOL_CHECK_FORMAT_STR |
    PMEMPOOL_CHECK_REPAIR |
    PMEMPOOL_CHECK_VERBOSE;

  if((ppc = pmempool_check_init(&args, sizeof(args))) == NULL) {
    perror("pmempool_check_init");
    return -1;
  }

  /* perform check and repair, answer 'yes' for each question */
  while ((status = pmempool_check(ppc)) != NULL) {
    switch (status->type) {
    case PMEMPOOL_CHECK_MSG_TYPE_ERROR:
    case PMEMPOOL_CHECK_MSG_TYPE_INFO:
      break;
    case PMEMPOOL_CHECK_MSG_TYPE_QUESTION:
      printf("%s\n", status->str.msg);
      status->str.answer = "yes";
      break;
    default:
      pmempool_check_end(ppc);
      return 1;
    }
  }

  /* finalize the check and get the result */
  int ret = pmempool_check_end(ppc);
  switch (ret) {
  case PMEMPOOL_CHECK_RESULT_CONSISTENT:
  case PMEMPOOL_CHECK_RESULT_REPAIRED:
    return 0;
  }

  return 1;
}

std::mutex pmempool_mutex; /*< global lock for non-thread safe pmempool_rm */

inline int safe_pmempool_rm(const char *path, int flags)
{
  std::lock_guard<std::mutex> g(pmempool_mutex);
  return pmempool_rm(path, flags);
}


/**------------------------------------------------------------------------- 
 * Main APIs
 * 
 */
struct root_table_t
{
  //TOID(struct hashmap_tx) map;
};
TOID_DECLARE_ROOT(struct root_table_t);

namespace nupm
{

void * allocate_dax_subregion(const std::string& device_path, void * hint)
{
  PMEMobjpool *pop = nullptr;
  
  /* open or re-create a pmemobj pool on this dax device */
  if(!check_pool(device_path.c_str())) {
    /* probably device dax */
    if(safe_pmempool_rm(device_path.c_str(), PMEMPOOL_RM_FORCE | PMEMPOOL_RM_POOLSET_LOCAL))
        throw General_exception("pmempool_rm on (%s) failed", device_path.c_str());
    PLOG("pmempool rm OK");
    pop = pmemobj_create(device_path.c_str(), REGION_NAME, 0, 0666);
  }
  else {
    PLOG("Pool was OK!");
  }
  PLOG("Pool pop=%p", pop);
  return nullptr;
}

} // namespace nupm

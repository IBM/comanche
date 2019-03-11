#ifndef __YCSB_DB_FACT_H__
#define __YCSB_DB_FACT_H__

#include "dawndb.h"
#include "db.h"
#include "properties.h"

namespace ycsb
{
class DBFactory {
 public:
  static DB* create(Properties& props, unsigned core = 0)
  {
    if (props.getProperty("db") == "dawn") {
      return new DawnDB(props, core);
    }
    else {
      return nullptr;
    }
  }
};

}  // namespace ycsb

#endif

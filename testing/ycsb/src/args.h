#ifndef __YCSB_ARGS_H__
#define __YCSB_ARGS_H__

#include "db.h"
#include "properties.h"

class Args {
 public:
  Args(ycsb::DB *db, Properties &props) : db(db), props(props) {}
  ycsb::DB *  db;
  Properties &props;
};

#endif

#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include "../src/db.h"
#include "../src/db_fact.h"
#include "../src/properties.h"
#include "../src/workload.h"

int main(){
   Properties props;
   props.setProperty("db", "dawn");
   props.setProperty("address", "10.0.0.71:11912");
   props.setProperty("dev", "mlx5_0");
   props.setProperty("recordcount", "10");
   props.setProperty("operationcount", "10");
   //   props.setProperty("requestdistribution", "uniform");
   props.setProperty("requestdistribution", "zipfian");
   // props.setProperty("readproportion", "1");
   props.setProperty("updateproportion", "1");
   ycsb::DB *db = ycsb::DBFactory::create(props);
   assert(db);
   ycsb::Workload* wl = new ycsb::Workload(props, db);
   assert(wl);
   wl->load();
   wl->run();
   delete wl;
   delete db;
}

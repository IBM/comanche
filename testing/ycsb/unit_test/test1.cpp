/*
   Copyright [2017-2019] [IBM Corporation]
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/
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

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


/* 
 * Author: Feng Li
 * e-mail: fengggli@yahoo.com
 *
 */

/*
 * the transaction support. refer to exampleslibpmemobj/map/mapcli.c
 */
#include <gtest/gtest.h>

#include "bitmap-tx.h"

#include <common/logging.h>
#include <common/exceptions.h>


static PMEMobjpool *pop;
TOID(struct bitmap_tx) map;
static char input[80];

static unsigned nbits = 4096;

#define TOO_LARGE (10000)

TEST(bitmap_test, init_objpool){
  int ret;
  PINF("input the obj pool name:");
  ret = scanf("%s", input);
  PINF("\nnow creating  obj pool [%s], ret =%d ", input, ret);

  pop = pmemobj_create(input, POBJ_LAYOUT_NAME(bitmap_store),
          PMEMOBJ_MIN_POOL, 0666);
  if ((pop) == NULL) {

      pop = pmemobj_open(input, POBJ_LAYOUT_NAME(bitmap_store));
			PINF("using existing pool \n");
      if(pop == NULL)
        throw General_exception("failed to re-open pool - %s\n", pmemobj_errormsg());
		}
}

TEST(bitmap_test, create){
  map = POBJ_ROOT(pop, struct bitmap_tx); 
  bitmap_tx_create(pop, map, nbits);
}

/*
 * small regions(less than a word)
 */
TEST(bitmap_test, small){
  // i should find 4096/16 = 256
  int nr_regions = 0;

  unsigned int pos;

  while(1){
    pos = bitmap_tx_find_free_region(pop, map, 2);
    if(pos == -1){
      PLOG("stop\n");
      break;
    }else{
      nr_regions +=1;
      //PDBG("allocate at pos %u", pos);
      PDBG("now %d instances", nr_regions);
    }


    if(nr_regions > TOO_LARGE){
      PERR("too mange instances");
      FAIL();
    }
  }

  ASSERT_EQ(1024, nr_regions);

  PINF("totally %d instances\n", nr_regions);
}


TEST(bitmap_test, zero){
  bitmap_tx_zero(pop, map);
  PINF("zeroing complete");
}

/* large regions
 */
TEST(bitmap_test, large){
  // i should find 4096/16 = 256
  int nr_regions = 0;

  unsigned int pos;

  while(1){
    pos = bitmap_tx_find_free_region(pop, map, 4);
    if(pos == -1){
      PLOG("stop\n");
      break;
    }else{
      nr_regions +=1;
      //PDBG("allocate at pos %u", pos);
      //PDBG("now %d instances", nr_regions);
    }

    if(nr_regions > TOO_LARGE){
      PERR("too mange instances");
      FAIL();
    }
  }

  ASSERT_EQ(256, nr_regions);

  PINF("totally %d instances\n", nr_regions);
}

TEST(bitmap_test, destory)
{
  // free the bitmap
  if(bitmap_tx_destroy( pop,  map)){
      throw General_exception("failed to release bitmap - %s\n", pmemobj_errormsg());
      };
  PINF("bitmap freed");
}



TEST(bitmap_test, close){
	pmemobj_close(pop);
  PINF("pmemobj closed");
}

int main(int argc, char **argv) {
  // TODO: relead the program and read the full bitmap again.
  
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}



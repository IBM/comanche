#ifndef __YCSB_DB_H__
#define __YCSB_DB_H__

#include <map>
#include <vector>
#include "properties.h"

using namespace ycsbutils;
using namespace std;

namespace ycsb{
	class DB{
		public:
                 DB(Properties &props) { init(props); }
                 virtual int  get(const string &pool,
                                  const string &key,
                                  char *        value,
                                  bool          direct = false) = 0;
                 virtual int  put(const string &pool,
                                  const string &key,
                                  const char *  value,
                                  bool          direct = false) = 0;
                 virtual int  update(const string &pool,
                                     const string &key,
                                     const char *  value,
                                     bool          direct = false)                  = 0;
                 virtual int  erase(const string &pool, const string &key) = 0;
                 virtual int  scan(const string &pool,
                                   const string &key,
                                   int           count,
                                   vector < map<string, string> & results) = 0;
                 ~DB()
                 {
                   clean();
                   delete this;
                 }

                private:
                 virtual void init(Properties &props) {}
                 virtual void clean() {}
        };

        class DBFactory {
         public:
          static DB *create(Properties &prop)
          {
            if (props["db"] == "dawn") {
              return new DawnDB();
            }
          }
        };
}

#endif

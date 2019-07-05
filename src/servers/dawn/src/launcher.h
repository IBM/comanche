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
#ifndef __DAWN_LAUNCHER_H__
#define __DAWN_LAUNCHER_H__

#include <common/logging.h>
#include <sstream>
#include <string>

#include "config_file.h"
#include "program_options.h"
#include "shard.h"

namespace Dawn
{
class Shard_launcher : public Config_file {
 public:
  Shard_launcher(Program_options& options) : Config_file(options.config_file)
  {
    for (unsigned i = 0; i < shard_count(); i++) {
      PMAJOR("launching shard: core(%d) port(%d) net(%s)", get_shard_core(i),
             get_shard_port(i), get_shard("net", i).c_str());

      auto dax_config = get_shard_dax_config(i);
      std::string dax_config_json;

      /* handle DAX config if needed */
      if(dax_config.size() > 0) {
        std::stringstream ss;
        ss << "[{\"region_id\":0,\"path\":\"" << dax_config[0].first
           << "\",\"addr\":\"" << dax_config[0].second << "\"}]";
        PLOG("DAX config %s", ss.str().c_str());
        dax_config_json = ss.str();
      }
      
      try
      {
        _shards.push_back(new Dawn::Shard(
            get_shard_core(i),
            get_shard_port(i), get_net_providers(),
            get_shard("device", i),
            get_shard("net", i),
            get_shard("default_backend", i),
            get_shard("index", i),
            get_shard("nvme_device", i),
            get_shard("pm_path", i),
            dax_config_json,
            get_shard("default_ado_path", i),
            options.debug_level,
            options.forced_exit));
      }
      catch (const std::exception &e)
      {
        PLOG("shard %d failed to launch: %s", i, e.what());
      }
    }
  }

  ~Shard_launcher()
  {
    PLOG("exiting shard (%p)", this);
    for (auto& sp : _shards) delete sp;
  }

  void wait_for_all()
  {
    bool alive;
    do {
      sleep(1);
      alive = false;
      for (auto& sp : _shards) {
        alive = !sp->exited();
      }
    } while (alive);
  }

 private:
  std::vector<Dawn::Shard*> _shards;
};
}  // namespace Dawn

#endif  // __DAWN_LAUNCHER_H__

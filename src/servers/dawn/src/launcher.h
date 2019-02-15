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
  static constexpr auto DEFAULT_PROVIDER = "verbs";

 public:
  Shard_launcher(Program_options& options) : Config_file(options.config_file)
  {
    for (unsigned i = 0; i < shard_count(); i++) {
      PMAJOR("launching shard: core(%d) port(%d) net(%s)", get_shard_core(i),
             get_shard_port(i), get_shard("net", i).c_str());

      auto dax_config = get_shard_dax_config(i);
      assert(dax_config.size() == 1);
      std::stringstream ss;
      ss << "[{\"region_id\":0,\"path\":\"" << dax_config[0].first
         << "\",\"addr\":\"" << dax_config[0].second << "\"}]";
      PLOG("DAX config %s", ss.str().c_str());
      const std::string dax_config_json = ss.str();

      _shards.push_back(new Dawn::Shard(
          get_shard_core(i), get_shard_port(i), DEFAULT_PROVIDER,
          get_shard("device", i), get_shard("net", i),
          get_shard("default_backend", i), get_shard("nvme_device", i),
          get_shard("pm_path", i), dax_config_json, options.debug_level,
          options.forced_exit));
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

#ifndef __DAWN_LAUNCHER_H__
#define __DAWN_LAUNCHER_H__

#include <common/logging.h>
#include <string>

#include "config_file.h"
#include "program_options.h"
#include "shard.h"

namespace Dawn
{
class Shard_launcher : public Config_file {
  static constexpr auto DEFAULT_PROVIDER = "verbs";

 public:
  Shard_launcher(Program_options& options) : Config_file(options.config_file) {
    for (unsigned i = 0; i < shard_count(); i++) {
      PMAJOR("launching shard: core(%d) port(%d) net(%s)", get_shard_core(i),
             get_shard_port(i), get_shard("net", i).c_str());

      _shards.push_back(new Dawn::Shard(
          get_shard_core(i), get_shard_port(i), DEFAULT_PROVIDER,
          get_shard("device", i), get_shard("net", i),
          get_shard("default_backend", i), get_shard("nvme_device", i),
          get_shard("pm_path", i), options.debug_level, options.forced_exit));
    }
  }

  ~Shard_launcher() {
    PLOG("exiting shard (%p)", this);
    for (auto& sp : _shards) delete sp;
  }

  void wait_for_all() {
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
}

#endif  // __DAWN_LAUNCHER_H__

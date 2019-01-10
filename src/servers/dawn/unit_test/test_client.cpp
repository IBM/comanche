#include <api/rdma_itf.h>
#include <common/logging.h>
#include <common/utils.h>
#include <dawn.h>
#include <stdlib.h>
#include <boost/program_options.hpp>

int main(int argc, char* argv[]) {
  int port;
  std::string server_addr;

  try {
    namespace po = boost::program_options;
    po::options_description desc("Options");
    desc.add_options()("help", "Show help")("port", po::value<unsigned>(), "Port to connect to")(
        "endpoint", po::value<std::string>(), "Endpoint name (IPC)")("addr", po::value<std::string>(), "Server IP address");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if (vm.count("help"))  // || vm.count("endpoint")==0 )
      std::cout << desc;

    port = vm.count("port") ? vm["port"].as<unsigned>() : 11911;
    server_addr = vm.count("addr") ? vm["addr"].as<std::string>() : "10.0.0.11";
    PLOG("using port: %u", port);
  }
  catch (...) {
    return -1;
  }

  using namespace Dawn::Client;

  Session s(server_addr.c_str(), port);

  auto p = s.open_pool("myPoolname", MB(32), Session::FLAGS_CREATE_ON_MISSING);
  PINF("Opened pool: %lx", p);

  std::string key = "Animal";
  std::string value = "Elephant";

  s.put(p, key, value);
  value.clear();

  s.get(p, key, value);
  PLOG("Woohoo: got value (%s)", value.c_str());
  s.shutdown();

  return 0;
}

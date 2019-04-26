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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <api/components.h>
#pragma GCC diagnostic pop

#include <gtest/gtest.h>

#pragma GCC diagnostic push
#if __clang__
#pragma GCC diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif
#include <common/logging.h>
#pragma GCC diagnostic pop

#include <api/fabric_itf.h>

#include "eyecatcher.h"
#include "patience.h" /* open_connection_patiently */
#include "pingpong_client.h"
#include "pingpong_server.h"
#include "pingpong_server_n.h"
#include "pingpong_stat.h"
#include "registration.h"
#include "remote_memory_server.h"
#include "remote_memory_server_grouped.h"
#include "remote_memory_subserver.h"
#include "server_grouped_connection.h"
#include "remote_memory_client_grouped.h"
#include "remote_memory_subclient.h"
#include "remote_memory_client.h"
#include "remote_memory_client_for_shutdown.h"

#include <sys/time.h>
#include <sys/resource.h>

#include <algorithm> /* max, min */
#include <chrono> /* seconds */
#include <cinttypes> /* PRIu64 */
#include <cstring> /* strpbrk */
#include <exception>
#include <stdexcept> /* domain_error */
#include <memory> /* shared_ptr */
#include <iostream> /* cerr */
#include <future> /* async, future */
#include <thread> /* sleep_for */

// The fixture for testing class Foo.
class Fabric_test : public ::testing::Test {

  static const char *control_port_spec;
 protected:

  // If the constructor and destructor are not enough for setting up
  // and cleaning up each test, you can define the following methods:

  virtual void SetUp() {
    // Code here will be called immediately after the constructor (right
    // before each test).
  }

  virtual void TearDown() {
    // Code here will be called immediately after each test (right
    // before the destructor).
  }

  // Objects declared here can be used by all tests in the test case

  static const std::uint16_t control_port_0;
  static const std::uint16_t control_port_1;
  static const std::uint16_t control_port_2;

  static std::string json_kv_pair(const std::string &key_, const std::string &value_)
  {
    return json_quote(key_) + " : " + value_ + "";
  }
  static std::string json_quote(const std::string &s_)
  {
    return "\"" + s_ + "\"";
  }
  static std::string fabric_spec(const std::string &prov_name_) {
    const std::string verbs_mr_mode =
      "[ " + json_quote("FI_MR_LOCAL") + "," + json_quote("FI_MR_VIRT_ADDR") + "," + json_quote("FI_MR_ALLOCATED") + "," + json_quote("FI_MR_PROV_KEY") + " ]";

    /* Although man fi_mr says "FI_MR_BASIC is maintained for backwards
     * compatibility (libfabric version 1.4 or earlier)", sockets as of 1.6
     * will not accept the newer, explicit list.
     *
     * Although man fi_mr says "providers that support basic registration
     * usually required FI_MR_LOCAL", the socket provider will not accept
     * FI_MR_LOCAL.
     */
    const std::string sockets_mr_mode =
      "[ " + json_quote("FI_MR_BASIC") + " ]";

	const std::string domain_name_verbs_spec =
      domain_name_verbs
      ? ", " + json_kv_pair("name", json_quote(domain_name_verbs))
      : std::string()
      ;

	const std::string domain_name_sockets_spec =
      domain_name_sockets
      ? ", " + json_kv_pair("name", json_quote(domain_name_sockets))
      : std::string()
      ;

	const std::string domain_name_spec =
      prov_name_ == std::string("sockets") ? domain_name_sockets_spec : domain_name_verbs_spec;

    const std::string &mr_mode = prov_name_ == std::string("sockets") ? sockets_mr_mode : verbs_mr_mode;

    return
      "{ "
       + json_kv_pair("fabric_attr", " { " + json_kv_pair("prov_name", json_quote(prov_name_)) + " }" )
       + ","
       + json_kv_pair("domain_attr"
          , " { "
          + json_kv_pair("mr_mode", mr_mode)
          + domain_name_spec +
          "}" )
       + ","
       + json_kv_pair("ep_attr", " { " + json_kv_pair("type", json_quote("FI_EP_MSG")) + " }" )
       + " }"
      ;
  }
public:
  static char *domain_name_verbs;
  static char *domain_name_sockets;
};

const char *Fabric_test::control_port_spec = std::getenv("FABRIC_TEST_CONTROL_PORT");
const std::uint16_t Fabric_test::control_port_0 = uint16_t(control_port_spec ? std::strtoul(control_port_spec, 0, 0) : 47591);
const std::uint16_t Fabric_test::control_port_1 = uint16_t(control_port_0 + 1);
const std::uint16_t Fabric_test::control_port_2 = uint16_t(control_port_0 + 2);


char *Fabric_test::domain_name_verbs;
char *Fabric_test::domain_name_sockets;

std::ostream &describe_ep(std::ostream &o_, const Component::IFabric_server_factory &e_)
{
  return o_ << "provider '" << e_.get_provider_name() << "' max_message_size " << e_.max_message_size();
}

namespace
{
  bool is_client = false;
  bool is_server = false;
  char *remote_host = nullptr;
}

namespace
{

void instantiate_server(const std::string &fabric_spec_, uint16_t control_port_)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);

  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<Component::IFabric_factory *>(comp->query_interface(Component::IFabric_factory::iid())));
  auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec_));

  {
    auto srv1 = std::shared_ptr<Component::IFabric_server_factory>(fabric->open_server_factory("{}", control_port_));
    describe_ep(std::cerr << "InstantiateServer " << fabric_spec_ << "Endpoint 1 ", *srv1) << std::endl;
  }
  {
    auto srv2 = std::shared_ptr<Component::IFabric_server_factory>(fabric->open_server_factory("{}", control_port_));
    describe_ep(std::cerr << "InstantiateServer " << fabric_spec_ << " Endpoint 2 ", *srv2) << std::endl;
  }
  factory->release_ref();
}

static constexpr auto count_outer = 3U;
static constexpr auto count_inner = 3U;
static constexpr std::size_t memory_size = 4096;
static constexpr unsigned long iterations = 1000000;
static constexpr std::size_t msg_size = 1U << 6U;

void write_read_sequential(const std::string &fabric_spec_, uint16_t control_port_, bool force_error_)
{
  for ( auto iter0 = 0U; iter0 != count_outer; ++iter0 )
  {
    /* To avoid conflicts with server which are slow to shut down, use a different control port on every pass
     * But, what happens if a server is shut down (fi_shutdown) while a client is expecting to receive data?
     * Shouldn't the client see some sort of error?
     */
    auto control_port = std::uint16_t(control_port_ + iter0);
    /* create object instance through factory */
    Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                        Component::net_fabric_factory);
    ASSERT_TRUE(comp);

    auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<Component::IFabric_factory *>(comp->query_interface(Component::IFabric_factory::iid())));

    auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec_));
    if ( is_server )
    {
      std::cerr << "SERVER begin " << iter0 << " port " << control_port << std::endl;
      {
        auto remote_key_base = 0U;
        remote_memory_server server(*fabric, "{}", control_port, "", memory_size, remote_key_base);
        EXPECT_LT(0U, server.max_message_size());
      }
      std::cerr << "SERVER end " << iter0 << std::endl;
    }
    else
    {
      /* allow time for the server to listen before the client restarts.
       * For an unknown reason, client "connect" puts the port in "ESTABSLISHED"
       * state, causing server "bind" to fail with EADDRINUSE.
       */
      std::this_thread::sleep_for(std::chrono::milliseconds(2500));
      std::size_t msg_max(0U);
      for ( auto iter1 = 0U; iter1 != count_inner; ++iter1 )
      {
        std::cerr << "CLIENT begin " << iter0 << "." << iter1 << " port " << control_port << std::endl;

        /* In case the provider actually uses the remote keys which we provide, make them unique. */
        auto remote_key_index = iter1;
        /* Feed the client a good JSON spec */
        remote_memory_client client(*fabric, "{}", remote_host, control_port, memory_size, remote_key_index);

        /* expect that all max messages are greater than 0, and the same */
        EXPECT_LT(0U, client.max_message_size());
        if ( 0 == iter1 )
        {
          msg_max = client.max_message_size();
        }
        else
        {
          EXPECT_EQ(client.max_message_size(), msg_max);
        }

        /* Feed the server_factory some terrible text */
        std::string msg = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.";
        /* An ordinary client which tests RDMA to server memory */
        client.write(msg, force_error_);
        if ( ! force_error_ )
        {
          client.read_verify(msg);
        }
        /* client destructor sends FI_SHUTDOWN to server */
        std::cerr << "CLIENT end " << iter0 << "." << iter1 << std::endl;
      }

      /* In case the provider actually uses the remote keys which we provide, make them unique.
       * (At server shutdown time there are no other clients, so the shutdown client may use any value.)
       */
      auto remote_key_index = 0U;
      /* A special client to tell the server factory to shut down. Placed after other clients because the server apparently cannot abide concurrent clients. */
      remote_memory_client_for_shutdown client_shutdown(*fabric, "{}", remote_host, control_port, memory_size, remote_key_index);
      EXPECT_EQ(client_shutdown.max_message_size(), msg_max);
    }

    factory->release_ref();
  }
}

template <typename D>
  double double_seconds(D d_)
  {
    return std::chrono::duration_cast<std::chrono::duration<double>>(d_).count();
  }

std::pair<timeval, timeval> cpu_time()
{
  struct rusage ru;
  {
    auto rc = ::getrusage(RUSAGE_SELF, &ru);
    EXPECT_EQ(rc, 0);
  }
  return { ru.ru_utime, ru.ru_stime };
}

std::chrono::microseconds usec(const timeval &start, const timeval &stop)
{
  return
    ( std::chrono::seconds(stop.tv_sec) + std::chrono::microseconds(stop.tv_usec) )
    -
    ( std::chrono::seconds(start.tv_sec) + std::chrono::microseconds(start.tv_usec) )
    ;
}

void ping_pong(const std::string &fabric_spec_, std::uint16_t control_port_2_, unsigned thread_count_)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                    Component::net_fabric_factory);
  ASSERT_TRUE(comp);

  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<Component::IFabric_factory *>(comp->query_interface(Component::IFabric_factory::iid())));
  auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec_));
  auto control_port = std::uint16_t(control_port_2_);

  const std::size_t buffer_size = std::max(msg_size << 1U, memory_size);

  std::chrono::nanoseconds cpu_user;
  std::chrono::nanoseconds cpu_system;
  std::chrono::high_resolution_clock::duration t{};
  std::chrono::high_resolution_clock::duration start_stagger{};
  std::chrono::high_resolution_clock::duration stop_stagger{};
  std::uint64_t poll_count = 0U;
  if ( is_server )
  {
    std::cerr << "SERVER begin port " << control_port << std::endl;

    auto ep = std::unique_ptr<Component::IFabric_server_factory>(fabric->open_server_factory("{}", control_port));
    EXPECT_LT(0U, ep->max_message_size());
    std::vector<std::future<pingpong_stat>> servers;
    /* In case the provider actually uses the remote keys which we provide, make them unique. */
    auto cpu_start = cpu_time();
    for ( auto remote_key_base = 0U; remote_key_base != thread_count_; ++remote_key_base )
    {
      servers.emplace_back(
        std::async(
          std::launch::async
          , [&ep, buffer_size, remote_key_base]
            {
              pingpong_server server(*ep, buffer_size, remote_key_base, iterations, msg_size);
              return server.time();
            }
        )
      );
    }

    auto start_max = std::chrono::high_resolution_clock::time_point::min();
    auto stop_max = std::chrono::high_resolution_clock::time_point::min();
    auto start_min = std::chrono::high_resolution_clock::time_point::max();
    auto stop_min = std::chrono::high_resolution_clock::time_point::max();
    for ( auto &f : servers )
    {
      auto r = f.get();
      start_min = std::min(start_min, r.start());
      start_max = std::max(start_max, r.start());
      stop_min = std::min(stop_min, r.stop());
      stop_max = std::max(stop_max, r.stop());
      poll_count += r.poll_count();
    }
    auto cpu_stop = cpu_time();
    cpu_user = usec(cpu_start.first, cpu_stop.first);
    cpu_system = usec(cpu_start.second, cpu_stop.second);
    std::cerr << "SERVER end\n";

    t = stop_max - start_min;
    start_stagger = start_max - start_min;
    stop_stagger = stop_max - stop_min;
  }
  else
  {
    auto start_delay = std::chrono::seconds(3);
    /* allow time for the server to listen before the client restarts */
    std::this_thread::sleep_for(start_delay);

    std::cerr << "CLIENT begin port " << control_port << std::endl;
    std::vector<std::future<pingpong_stat>> clients;
    /* In case the provider actually uses the remote keys which we provide, make them unique. */
    auto cpu_start = cpu_time();
    for ( auto remote_key_base = 0U; remote_key_base != thread_count_; ++remote_key_base )
    {
      clients.emplace_back(
        std::async(
          std::launch::async
          , [&fabric, control_port, buffer_size, remote_key_base]
            {
              auto id = static_cast<std::uint8_t>(remote_key_base);
              pingpong_client client(*fabric, "{}", remote_host, control_port, buffer_size, remote_key_base, iterations, msg_size, id);
              return client.time();
            }
        )
      );
    }
    auto start_max = std::chrono::high_resolution_clock::time_point::min();
    auto stop_max = std::chrono::high_resolution_clock::time_point::min();
    auto start_min = std::chrono::high_resolution_clock::time_point::max();
    auto stop_min = std::chrono::high_resolution_clock::time_point::max();

    for ( auto &f : clients )
    {
      auto r = f.get();
      start_min = std::min(start_min, r.start());
      start_max = std::max(start_max, r.start());
      stop_min = std::min(stop_min, r.stop());
      stop_max = std::max(stop_max, r.stop());
      poll_count += r.poll_count();
    }
    auto cpu_stop = cpu_time();
    cpu_user = usec(cpu_start.first, cpu_stop.first);
    cpu_system = usec(cpu_start.second, cpu_stop.second);
    std::cerr << "CLIENT end\n";
    t = stop_max - start_min;
    start_stagger = start_max - start_min;
    stop_stagger = stop_max - stop_min;
  }

  auto iter = thread_count_ * iterations;
  auto iterf = static_cast<double>(iter);
  auto secs_inner = double_seconds(t);
  PINF("%zu byte PingPong, iterations/client: %lu clients: %u secs: %f start stagger: %f stop stagger: %f cpu_user: %f cpu_sys: %f Ops/Sec: %lu Polls/Op: %f"
    , msg_size
    , iterations
    , thread_count_
    , secs_inner
    , double_seconds(start_stagger)
    , double_seconds(stop_stagger)
    , double_seconds(cpu_user)
    , double_seconds(cpu_system)
    , static_cast<unsigned long>( iterf / secs_inner )
    , static_cast<double>(poll_count)/iterf
  );
  factory->release_ref();
}

void pingpong_single_server(const std::string &fabric_spec_, std::uint16_t control_port_2, unsigned client_count_)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                    Component::net_fabric_factory);
  ASSERT_TRUE(comp);

  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<Component::IFabric_factory *>(comp->query_interface(Component::IFabric_factory::iid())));
  auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec_));
  auto control_port = std::uint16_t(control_port_2);

  const std::size_t buffer_size = std::max(msg_size << 1U, memory_size);
  std::uint64_t poll_count = 0U;

  std::chrono::nanoseconds cpu_user;
  std::chrono::nanoseconds cpu_system;
  std::chrono::high_resolution_clock::duration t{};
  std::chrono::high_resolution_clock::duration start_stagger{};
  std::chrono::high_resolution_clock::duration stop_stagger{};

  if ( is_server )
  {
    std::cerr << "SERVER begin port " << control_port << std::endl;

    auto ep = std::unique_ptr<Component::IFabric_server_factory>(fabric->open_server_factory("{}", control_port));
    EXPECT_LT(0U, ep->max_message_size());

    /* In case the provider actually uses the remote keys which we provide, make them unique. */
    auto remote_key_base = 0U;
    auto cpu_start = cpu_time();
    pingpong_server_n server(client_count_, *ep, buffer_size, remote_key_base, iterations, msg_size);
    auto f2 = server.time();
    auto cpu_stop = cpu_time();
    cpu_user = usec(cpu_start.first, cpu_stop.first);
    cpu_system = usec(cpu_start.second, cpu_stop.second);
    t = f2.stop() - f2.start();
    poll_count += f2.poll_count();

    std::cerr << "SERVER end" << std::endl;
  }
  else
  {
    auto start_delay = std::chrono::seconds(3);
    /* allow time for the server to listen before the client restarts */
    std::this_thread::sleep_for(start_delay);

    std::cerr << "CLIENT begin port " << control_port << std::endl;
    std::vector<std::future<pingpong_stat>> clients;
    /* In case the provider actually uses the remote keys which we provide, make them unique. */
    auto cpu_start = cpu_time();
    for ( auto remote_key_base = 0U; remote_key_base != client_count_; ++remote_key_base )
    {
      clients.emplace_back(
        std::async(
          std::launch::async
          , [&fabric, control_port, buffer_size, remote_key_base]
            {
              auto id = static_cast<std::uint8_t>(remote_key_base);
              pingpong_client client(*fabric, "{}", remote_host, control_port, buffer_size, remote_key_base, iterations, msg_size, id);
              return client.time();
            }
        )
      );
    }
    auto start_max = std::chrono::high_resolution_clock::time_point::min();
    auto stop_max = std::chrono::high_resolution_clock::time_point::min();
    auto start_min = std::chrono::high_resolution_clock::time_point::max();
    auto stop_min = std::chrono::high_resolution_clock::time_point::max();

    for ( auto &f : clients )
    {
      auto r = f.get();
      start_min = std::min(start_min, r.start());
      start_max = std::max(start_max, r.start());
      stop_min = std::min(stop_min, r.stop());
      stop_max = std::max(stop_max, r.stop());
      poll_count += r.poll_count();
      std::cerr << "CLIENT end " << double_seconds(r.stop() - r.start()) << " sec" << std::endl;
    }
    auto cpu_stop = cpu_time();
    cpu_user = usec(cpu_start.first, cpu_stop.first);
    cpu_system = usec(cpu_start.second, cpu_stop.second);
    std::cerr << "CLIENT end" << std::endl;
    t = stop_max - start_min;
    start_stagger = start_max - start_min;
    stop_stagger = stop_max - stop_min;
  }

  auto iter = client_count_ * iterations;
  auto iterf = static_cast<double>(iter);
  auto secs_inner = double_seconds(t);

  PINF("%zu byte PingPong, iterations/client: %lu clients: %u secs: %f start stagger: %f stop stagger: %f cpu_user: %f cpu_sys: %f Ops/Sec: %lu Polls/Op: %f"
    , msg_size
    , iterations
    , client_count_
    , secs_inner
    , double_seconds(start_stagger)
    , double_seconds(stop_stagger)
    , double_seconds(cpu_user)
    , double_seconds(cpu_system)
    , static_cast<unsigned long>( iterf / secs_inner )
    , static_cast<double>(poll_count)/iterf
);
  factory->release_ref();
}

void instantiate_server_dual(const std::string &fabric_spec_, uint16_t control_port_0_, uint16_t control_port_1_)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);

  ASSERT_TRUE(comp);
  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<Component::IFabric_factory *>(comp->query_interface(Component::IFabric_factory::iid())));
  auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec_));

  try {
    /* fails, because both servers use the same port */
    auto srv1 = std::shared_ptr<Component::IFabric_server_factory>(fabric->open_server_factory("{}", control_port_0_));
    auto srv2 = std::shared_ptr<Component::IFabric_server_factory>(fabric->open_server_factory("{}", control_port_1_));
    describe_ep(std::cerr << "ISD Endpoint 1 ", *srv1) << std::endl;
    describe_ep(std::cerr << "ISD Endpoint 2 ", *srv2) << std::endl;
  }
  catch ( std::exception & )
  {
  }
  factory->release_ref();
}

TEST_F(Fabric_test, InstantiateServer)
{
  instantiate_server(fabric_spec("verbs"), control_port_0);
}

TEST_F(Fabric_test, InstantiateServer_Socket)
{
  instantiate_server(fabric_spec("sockets"), control_port_0);
}

TEST_F(Fabric_test, InstantiateServerDual)
{
  instantiate_server_dual(fabric_spec("verbs"), control_port_0, control_port_1);
}

TEST_F(Fabric_test, InstantiateServerDual_Sockets)
{
  instantiate_server_dual(fabric_spec("sockets"), control_port_0, control_port_1);
}

TEST_F(Fabric_test, JsonSucceed)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);

  ASSERT_TRUE(comp);
  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<Component::IFabric_factory *>(comp->query_interface(Component::IFabric_factory::iid())));
  auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec("verbs")));
  /* Feed the server_factory a good JSON spec */

  try
  {
    auto pep1 = std::shared_ptr<Component::IFabric_server_factory>(fabric->open_server_factory("{ \"tx_attr\" : { \"comp_order\" : [ \"FI_ORDER_STRICT\", \"FI_ORDER_DATA\" ], \"inject_size\" : 16 } }", control_port_1));
    describe_ep(std::cerr << "Endpoint 1 ", *pep1) << std::endl;
  }
  catch ( const std::domain_error &e )
  {
    std::cerr << "Unexpected exception: " << e.what() << std::endl;
    EXPECT_TRUE(false);
  }
  factory->release_ref();
}

TEST_F(Fabric_test, JsonParseAddrStr)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);

  ASSERT_TRUE(comp);
  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<Component::IFabric_factory *>(comp->query_interface(Component::IFabric_factory::iid())));
  auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec("verbs")));

  {
    /* Feed the ednpoint a good JSON spec */
    try
    {
      auto pep1 = std::shared_ptr<Component::IFabric_server_factory>(fabric->open_server_factory(("{ \"addr_format\" : \"FI_ADDR_STR\", \"dest\" : \"fi_shm://" + std::to_string(getpid()) + "\" }").c_str(), control_port_1));
    }
    catch ( const std::exception &e )
    {
      std::cerr << "Unexpected exception: " << e.what() << std::endl;
      EXPECT_TRUE(false);
    }
  }
  factory->release_ref();
}

TEST_F(Fabric_test, JsonParseFail)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);

  ASSERT_TRUE(comp);
  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<Component::IFabric_factory *>(comp->query_interface(Component::IFabric_factory::iid())));
  auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec("verbs")));

  {
    /* Feed the ednpoint a good JSON spec */
    try
    {
      auto pep1 = std::shared_ptr<Component::IFabric_server_factory>(fabric->open_server_factory("{ \"tx_attr\" : { \"max\" : \"xyz\"} } }", control_port_1));
      EXPECT_TRUE(false);
    }
    catch ( const std::domain_error &e )
    {
      /* Error mesage should mention "parse" */
      EXPECT_TRUE(::strpbrk(e.what(), "parse"));
    }
  }
  factory->release_ref();
}

TEST_F(Fabric_test, JsonKeyFail)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);

  ASSERT_TRUE(comp);
  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<Component::IFabric_factory *>(comp->query_interface(Component::IFabric_factory::iid())));
  auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec("verbs")));

  {
    /* Feed the ednpoint a good JSON spec */
    try
    {
      auto pep1 = std::shared_ptr<Component::IFabric_server_factory>(fabric->open_server_factory("{ \"tx_attr\" : { \"maX\" : \"xyz\"} }", control_port_1));
      EXPECT_TRUE(false);
    }
    catch ( const std::domain_error &e )
    {
      /* Error message should mention "key", "tx_attr", and "maX" */
      EXPECT_TRUE(::strpbrk(e.what(), "key"));
      EXPECT_TRUE(::strpbrk(e.what(), "tx_attr"));
      EXPECT_TRUE(::strpbrk(e.what(), "maX"));
    }
  }
  factory->release_ref();
}

void instantiate_server_and_client(const std::string &fabric_spec_, std::uint16_t control_port)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);

  ASSERT_TRUE(comp);
  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<Component::IFabric_factory *>(comp->query_interface(Component::IFabric_factory::iid())));
  auto fabric0 = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec_));
  auto fabric1 = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec_));

  {
    /* Feed the server_factory a good JSON spec */
    auto server = std::shared_ptr<Component::IFabric_server_factory>(fabric0->open_server_factory(fabric_spec_, control_port));
    EXPECT_LT(0U, server->max_message_size());
    auto client = std::shared_ptr<Component::IFabric_client>(open_connection_patiently(*fabric1, fabric_spec_, "127.0.0.1", control_port));
    EXPECT_LT(0U, client->max_message_size());
    EXPECT_EQ(server->max_message_size(), client->max_message_size());
  }

  factory->release_ref();
}

TEST_F(Fabric_test, InstantiateServerAndClient)
{
  instantiate_server_and_client(fabric_spec("verbs"), control_port_1);
}

TEST_F(Fabric_test, InstantiateServerAndClientSockets)
{
  instantiate_server_and_client(fabric_spec("sockets"), control_port_1);
}

TEST_F(Fabric_test, WriteReadSequential)
{
  write_read_sequential(fabric_spec("verbs"), control_port_2, false);
}

TEST_F(Fabric_test, WriteReadSequentialSockets)
{
  write_read_sequential(fabric_spec("sockets"), control_port_2, false);
}

TEST_F(Fabric_test, WriteReadSequentialWithError)
{
  write_read_sequential(fabric_spec("sockets"), control_port_2, true);
}

TEST_F(Fabric_test, WriteReadParallel)
{
  for ( auto iter0 = 0U; iter0 != count_outer; ++iter0 )
  {
    /* To avoid conflicts with server which are slow to shut down, use a different control port on every pass
     * But, what happens if a server is shut down (fi_shutdown) while a client is expecting to receive data?
     * Shouldn't the client see some sort of error?
     */
    auto control_port = std::uint16_t(control_port_2 + iter0);
    /* create object instance through factory */
    Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                        Component::net_fabric_factory);
    ASSERT_TRUE(comp);

    auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<Component::IFabric_factory *>(comp->query_interface(Component::IFabric_factory::iid())));

    auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec("verbs")));
    if ( is_server )
    {
      std::cerr << "SERVER begin " << iter0 << " port " << control_port << std::endl;
      {
        auto expected_client_count = count_inner;
        auto remote_key_base = 0U;
        remote_memory_server server(*fabric, "{}", control_port, "", memory_size, remote_key_base, expected_client_count);
        EXPECT_LT(0U, server.max_message_size());
      }
      std::cerr << "SERVER end " << iter0 << std::endl;
    }
    else
    {
      /* allow time for the server to listen before the client restarts.
       * For an unknown reason, client "connect" puts the port in "ESTABSLISHED"
       * state, causing server "bind" to fail with EADDRINUSE.
       */
      std::this_thread::sleep_for(std::chrono::milliseconds(2500));
      {
        std::vector<remote_memory_client> vv;
        for ( auto iter1 = 0U; iter1 != count_inner; ++iter1 )
        {
          /* In case the provider actually uses the remote keys which we provide, make them unique. */
          auto remote_key_index = iter1;

          /* Ordinary clients which test RDMA to server memory.
           * Should be able to control them via pointers or (using move semantics) in a vector of objects.
           */
          vv.emplace_back(*fabric, "{}", remote_host, control_port, memory_size, remote_key_index);
          /* expect that all max messages are greater than 0, and the same */
          EXPECT_LT(0U, vv.back().max_message_size());
          EXPECT_EQ(vv.front().max_message_size(), vv.back().max_message_size());
        }

        /* Feed the server_factory some terrible text */
        std::string msg = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.";

        for ( auto &client : vv )
        {
          client.write(msg);
        }

        for ( auto &client : vv )
        {
          client.read_verify(msg);
        }
        /* client destructors send FI_SHUTDOWNs to server */
      }
      /* The remote_memory servr will shut down after it has seen a specified number of clients. */
    }

    factory->release_ref();
  }
}

TEST_F(Fabric_test, GroupedClients)
{
  for ( uint16_t iter0 = 0U; iter0 != count_outer; ++iter0 )
  {
    /* create object instance through factory */
    Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);
    ASSERT_TRUE(comp);

    auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<Component::IFabric_factory *>(comp->query_interface(Component::IFabric_factory::iid())));

    auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec("verbs")));

    /* To avoid conflicts with server which are slow to shut down, use a different control port on every pass
     * But, what happens if a server is shut down (fi_shutdown) while a client is expecting to receive data?
     * Shouldn't the client see some sort of error?
     */
    auto control_port = std::uint16_t(control_port_2 + iter0);

    if ( is_server )
    {
      std::cerr << "SERVER begin " << iter0 << std::endl;
      {
        auto remote_key_base = 0U;
        remote_memory_server server(*fabric, "{}", control_port, "", memory_size, remote_key_base);
        EXPECT_LT(0U, server.max_message_size());
      }
      std::cerr << "SERVER end " << iter0 << std::endl;
    }
    else
    {
      /* allow time for the server to listen before the client restarts */
      std::this_thread::sleep_for(std::chrono::seconds(3));
      std::size_t msg_max(0U);
      for ( unsigned iter1 = 0U; iter1 != count_inner; ++iter1 )
      {
        std::cerr << "CLIENT begin " << iter0 << "." << iter1 << " port " << control_port << std::endl;
        /* In case the provider actually uses the remote keys which we provide, make them unique. */
        auto remote_key_index = iter1 * 3U;

        remote_memory_client_grouped client(*fabric, "{}", remote_host, control_port, memory_size, remote_key_index);

        /* expect that all max messages are greater than 0, and the same */
        EXPECT_LT(0U, client.max_message_size());
        if ( 0 == iter1 )
        {
          msg_max = client.max_message_size();
        }
        else
        {
          EXPECT_EQ(client.max_message_size(), msg_max);
        }

        /* make two communicators (three, including the parent. */
        remote_memory_subclient g0(client, memory_size, remote_key_index + 1U);
        remote_memory_subclient g1(client, memory_size, remote_key_index + 2U);

        std::string msg = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.";
        /* ought to split send from completions so that we can test the separation of comms */
        g0.write(msg);
        g0.read_verify(msg);
        g1.write(msg);
        g1.read_verify(msg);

        /* client destructor sends FI_SHUTDOWN to server */
        std::cerr << "CLIENT end " << iter0 << "." << iter1 << std::endl;
      }

      /* In case the provider actually uses the remote keys which we provide, make them unique.
       * (At server shutdown time there are no other clients, so the shutdown client may use any value.)
       */
      auto remote_key_index = 0U;
      /* A special client to tell the server to shut down. Placed after other clients because the server apparently cannot abide concurrent clients. */
      remote_memory_client_for_shutdown client_shutdown(*fabric, "{}", remote_host, control_port, memory_size, remote_key_index);
      EXPECT_EQ(client_shutdown.max_message_size(), msg_max);
    }

    factory->release_ref();
  }
}

TEST_F(Fabric_test, GroupedServer)
{
  for ( uint16_t iter0 = 0U; iter0 != count_outer; ++iter0 )
  {
    /* create object instance through factory */
    Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);
    ASSERT_TRUE(comp);

    auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<Component::IFabric_factory *>(comp->query_interface(Component::IFabric_factory::iid())));

    auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec("verbs")));

    /* To avoid conflicts with server which are slow to shut down, use a different control port on every pass
     * But, what happens if a server is shut down (fi_shutdown) while a client is expecting to receive data?
     * Shouldn't the client see some sort of error?
     */
    auto control_port = std::uint16_t(control_port_2 + iter0);

    if ( is_server )
    {
      std::cerr << "SERVER begin " << iter0 << std::endl;
      {
        auto remote_key_base = 0U;
        remote_memory_server_grouped server(*fabric, "{}", control_port, memory_size, remote_key_base);
        /* The server needs one permanent communicator, to handle the client
         * "disconnect" message. It does not need any other communicators,
         * but if it did, then the server (created by remote_memory_server_grouped
         * when it sees a client) would be the entity to create them.
         */
        EXPECT_LT(0U, server.max_message_size());
      }
      std::cerr << "SERVER end " << iter0 << std::endl;
    }
    else
    {
      /* allow time for the server to listen before the client restarts */
      std::this_thread::sleep_for(std::chrono::seconds(3));
      for ( auto iter1 = 0U; iter1 != count_inner; ++iter1 )
      {
        /* In case the provider actually uses the remote keys which we provide, make them unique. */
        auto remote_key_index = iter1;

        std::cerr << "CLIENT begin " << iter0 << "." << iter1 << " port " << control_port << std::endl;
        remote_memory_client client(*fabric, "{}", remote_host, control_port, memory_size, remote_key_index);

        std::string msg = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.";
        /* ought to split send from completions so that we can test the separation of comms */
        client.write(msg);
        client.read_verify(msg);

        /* client destructor sends FI_SHUTDOWN to server */
        std::cerr << "CLIENT end " << iter0 << "." << iter1 << std::endl;
      }

      /* In case the provider actually uses the remote keys which we provide, make them unique.
       * (At server shutdown time there are no other clients, so the shutdown client may use any value.)
       */
      auto remote_key_index = 0U;
      /* A special client to tell the server to shut down. Placed after other clients because the server apparently cannot abide concurrent clients. */
      remote_memory_client_for_shutdown client_shutdown(*fabric, "{}", remote_host, control_port, memory_size, remote_key_index);
    }

    factory->release_ref();
  }
}

TEST_F(Fabric_test, PingPong_1Threads)
{
  ping_pong(fabric_spec("verbs"), control_port_2, 1U);
  ping_pong(fabric_spec("verbs"), control_port_2, 1U);
}

TEST_F(Fabric_test, PingPong_2Threads)
{
  ping_pong(fabric_spec("verbs"), control_port_2, 2U);
  ping_pong(fabric_spec("verbs"), control_port_2, 2U);
}

TEST_F(Fabric_test, PingPong_4Threads)
{
  ping_pong(fabric_spec("verbs"), control_port_2, 4U);
  ping_pong(fabric_spec("verbs"), control_port_2, 4U);
}

TEST_F(Fabric_test, PingPong_8Threads)
{
  ping_pong(fabric_spec("verbs"), control_port_2, 8U);
  ping_pong(fabric_spec("verbs"), control_port_2, 8U);
}

TEST_F(Fabric_test, PingPong_16Threads)
{
  ping_pong(fabric_spec("verbs"), control_port_2, 16U);
  ping_pong(fabric_spec("verbs"), control_port_2, 16U);
}

TEST_F(Fabric_test, PingPong1Server_1Client)
{
  pingpong_single_server(fabric_spec("verbs"), control_port_2, 1U);
  pingpong_single_server(fabric_spec("verbs"), control_port_2, 1U);
}

TEST_F(Fabric_test, PingPong1Server_2Clients)
{
  pingpong_single_server(fabric_spec("verbs"), control_port_2, 2U);
  pingpong_single_server(fabric_spec("verbs"), control_port_2, 2U);
}

TEST_F(Fabric_test, PingPong1Server_4Clients)
{
  pingpong_single_server(fabric_spec("verbs"), control_port_2, 4U);
  pingpong_single_server(fabric_spec("verbs"), control_port_2, 4U);
}

TEST_F(Fabric_test, PingPong1Server_8Clients)
{
  pingpong_single_server(fabric_spec("verbs"), control_port_2, 8U);
  pingpong_single_server(fabric_spec("verbs"), control_port_2, 8U);
}

TEST_F(Fabric_test, PingPong1Server_16Clients)
{
  pingpong_single_server(fabric_spec("verbs"), control_port_2, 16U);
  pingpong_single_server(fabric_spec("verbs"), control_port_2, 16U);
}

} // namespace

int main(int argc, char **argv)
{
  is_client = bool(argv[1]);
  is_server = ! is_client;
  Fabric_test::domain_name_verbs = getenv("DOMAIN_VERBS");
  Fabric_test::domain_name_sockets = getenv("DOMAIN_SOCKETS");
  std::cerr << "Domain/verbs is " << (Fabric_test::domain_name_verbs ? Fabric_test::domain_name_verbs : "unspecified") << "\n";
  std::cerr << "Domain/sockets is " << (Fabric_test::domain_name_sockets ? Fabric_test::domain_name_sockets : "unspecified") << "\n";

  if ( is_client && argc != 2 )
  {
    PINF("%s [<ipaddr>]", argv[0]);
    return -1;
  }
  remote_host = argv[1];

  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}

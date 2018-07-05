/* note: we do not include component source, only the API definition */
#include <gtest/gtest.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <api/components.h>
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#if __clang__
#pragma GCC diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#endif
#include <common/logging.h>
#pragma GCC diagnostic pop

#include <api/fabric_itf.h>

#include "eyecatcher.h"
#include "patience.h" /* open_connection_patiently */
#include "registration.h"
#include "remote_memory_server.h"
#include "remote_memory_server_grouped.h"
#include "remote_memory_subserver.h"
#include "server_grouped_connection.h"
#include "remote_memory_client_grouped.h"
#include "remote_memory_subclient.h"
#include "remote_memory_client.h"
#include "remote_memory_client_for_shutdown.h"

#include <chrono> /* seconds */
#include <cstring> /* strpbrk */
#include <exception>
#include <stdexcept> /* domain_error */
#include <memory> /* shared_ptr */
#include <iostream> /* cerr */
#include <thread> /* sleep_for */

namespace {
  const std::string fabric_spec_static{
    "{ \"fabric_attr\" : { \"prov_name\" : \"verbs\" },"
    " \"domain_attr\" : "
      "{ \"mr_mode\" : [ \"FI_MR_LOCAL\", \"FI_MR_VIRT_ADDR\", \"FI_MR_ALLOCATED\", \"FI_MR_PROV_KEY\" ] }"
    ","
    " \"ep_attr\" : { \"type\" : \"FI_EP_MSG\" }"
    "}"
  };
}

// The fixture for testing class Foo.
class Fabric_test : public ::testing::Test {

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
  const std::string fabric_spec{fabric_spec_static};
  // const std::string fabric_spec{"{ \"fabric_attr\" : { \"prov_name\" : \"verbs\" } }"};
};

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
TEST_F(Fabric_test, InstantiateServer)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);

  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<Component::IFabric_factory *>(comp->query_interface(Component::IFabric_factory::iid())));
  auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec));

  {
    auto srv1 = std::shared_ptr<Component::IFabric_server_factory>(fabric->open_server_factory("{}", control_port_0));
    describe_ep(std::cerr << "InstantiateServer Endpoint 1 ", *srv1) << std::endl;
  }
  {
    auto srv2 = std::shared_ptr<Component::IFabric_server_factory>(fabric->open_server_factory("{}", control_port_0));
    describe_ep(std::cerr << "InstantiateServer Endpoint 2 ", *srv2) << std::endl;
  }
  factory->release_ref();
}

TEST_F(Fabric_test, InstantiateServerDual)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);

  ASSERT_TRUE(comp);
  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<Component::IFabric_factory *>(comp->query_interface(Component::IFabric_factory::iid())));
  auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec));

  try {
    /* fails, because both servers use the same port */
    auto srv1 = std::shared_ptr<Component::IFabric_server_factory>(fabric->open_server_factory("{}", control_port_0));
    auto srv2 = std::shared_ptr<Component::IFabric_server_factory>(fabric->open_server_factory("{}", control_port_1));
    describe_ep(std::cerr << "ISD Endpoint 1 ", *srv1) << std::endl;
    describe_ep(std::cerr << "ISD Endpoint 2 ", *srv2) << std::endl;
  }
  catch ( std::exception & )
  {
  }
  factory->release_ref();
}

TEST_F(Fabric_test, JsonSucceed)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);

  ASSERT_TRUE(comp);
  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<Component::IFabric_factory *>(comp->query_interface(Component::IFabric_factory::iid())));
  auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec));
  /* Feed the server_factory a good JSON spec */

  try
  {
    auto pep1 = std::shared_ptr<Component::IFabric_server_factory>(fabric->open_server_factory("{ \"tx_attr\" : { \"comp_order\" : [ \"FI_ORDER_STRICT\", \"FI_ORDER_DATA\" ], \"inject_size\" : 16 } }", control_port_1));
    describe_ep(std::cerr << "Endpoint 1 ", *pep1) << std::endl;
  }
  catch ( const std::domain_error &e )
  {
    std::cerr << "Unexpected exception: " << e.what() << std::endl;
    ASSERT_TRUE(false);
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
  auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec));

  {
    /* Feed the ednpoint a good JSON spec */
    try
    {
      auto pep1 = std::shared_ptr<Component::IFabric_server_factory>(fabric->open_server_factory(("{ \"addr_format\" : \"FI_ADDR_STR\", \"dest\" : \"fi_shm://" + std::to_string(getpid()) + "\" }").c_str(), control_port_1));
    }
    catch ( const std::exception &e )
    {
      std::cerr << "Unexpected exception: " << e.what() << std::endl;
      ASSERT_TRUE(false);
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
  auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec));

  {
    /* Feed the ednpoint a good JSON spec */
    try
    {
      auto pep1 = std::shared_ptr<Component::IFabric_server_factory>(fabric->open_server_factory("{ \"tx_attr\" : { \"max\" : \"xyz\"} } }", control_port_1));
      ASSERT_TRUE(false);
    }
    catch ( const std::domain_error &e )
    {
      /* Error mesage should mention "parse" */
      ASSERT_TRUE(::strpbrk(e.what(), "parse"));
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
  auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec));

  {
    /* Feed the ednpoint a good JSON spec */
    try
    {
      auto pep1 = std::shared_ptr<Component::IFabric_server_factory>(fabric->open_server_factory("{ \"tx_attr\" : { \"maX\" : \"xyz\"} }", control_port_1));
      ASSERT_TRUE(false);
    }
    catch ( const std::domain_error &e )
    {
      /* Error message should mention "key", "tx_attr", and "maX" */
      ASSERT_TRUE(::strpbrk(e.what(), "key"));
      ASSERT_TRUE(::strpbrk(e.what(), "tx_attr"));
      ASSERT_TRUE(::strpbrk(e.what(), "maX"));
    }
  }
  factory->release_ref();
}

TEST_F(Fabric_test, InstantiateServerAndClient)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);

  ASSERT_TRUE(comp);
  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<Component::IFabric_factory *>(comp->query_interface(Component::IFabric_factory::iid())));
  auto fabric0 = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec));
  auto fabric1 = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec));

  {
    /* Feed the server_factory a good JSON spec */
    auto server = std::shared_ptr<Component::IFabric_server_factory>(fabric0->open_server_factory(fabric_spec, control_port_1));
    auto client = std::shared_ptr<Component::IFabric_client>(open_connection_patiently(*fabric1, fabric_spec, "127.0.0.1", control_port_1));
  }

  factory->release_ref();
}

static constexpr auto count_outer = 3U;
static constexpr auto count_inner = 5U;

TEST_F(Fabric_test, WriteReadSequential)
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

    auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec_static));
    if ( is_server )
    {
      std::cerr << "SERVER begin " << iter0 << " port " << control_port << std::endl;
      {
        remote_memory_server server(*fabric, "{}", control_port);
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
      for ( auto iter1 = 0; iter1 != count_inner; ++iter1 )
      {
        std::cerr << "CLIENT begin " << iter0 << "." << iter1 << " port " << control_port << std::endl;

        /* Feed the client a good JSON spec */
        remote_memory_client client(*fabric, "{}", remote_host, control_port);
        /* Feed the server_factory some terrible text */
        std::string msg = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.";
        /* An ordinary client which tests RDMA to server memory */
        client.write(msg);
        client.read_verify(msg);
        /* client destructor sends FI_SHUTDOWN to server */
        std::cerr << "CLIENT end " << iter0 << "." << iter1 << std::endl;
      }

      /* A special client to tell the server factory to shut down. Placed after other clients because the server apparently cannot abide concurrent clients. */
      remote_memory_client_for_shutdown client_shutdown(*fabric, "{}", remote_host, control_port);
    }

    factory->release_ref();
  }
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

    auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec_static));
    if ( is_server )
    {
      std::cerr << "SERVER begin " << iter0 << " port " << control_port << std::endl;
      {
        auto expected_client_count = count_inner;
        remote_memory_server server(*fabric, "{}", control_port, expected_client_count);
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
        for ( auto iter1 = 0; iter1 != count_inner; ++iter1 )
        {
          /* Ordinary clients which test RDMA to server memory.
           * Should be able to control them via pointers or (using move semantics) in a vector of objects.
           */
          vv.emplace_back(*fabric, "{}", remote_host, control_port);
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

    auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec_static));

    /* To avoid conflicts with server which are slow to shut down, use a different control port on every pass
     * But, what happens if a server is shut down (fi_shutdown) while a client is expecting to receive data?
     * Shouldn't the client see some sort of error?
     */
    auto control_port = std::uint16_t(control_port_2 + iter0);

    if ( is_server )
    {
      std::cerr << "SERVER begin " << iter0 << std::endl;
      {
        remote_memory_server server(*fabric, "{}", control_port);
      }
      std::cerr << "SERVER end " << iter0 << std::endl;
    }
    else
    {
      /* allow time for the server to listen before the client restarts */
      std::this_thread::sleep_for(std::chrono::seconds(3));
      for ( auto iter1 = 0; iter1 != count_inner; ++iter1 )
      {
        std::cerr << "CLIENT begin " << iter0 << "." << iter1 << " port " << control_port << std::endl;
        remote_memory_client_grouped client(*fabric, "{}", remote_host, control_port);

        /* make one "communicator (two, including the parent. */
        remote_memory_subclient g0(client);
        remote_memory_subclient g1(client);
        std::string msg = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.";
        /* ought to split send from completions so that we can test the separation of comms */
        g0.write(msg);
        g0.read_verify(msg);
        g1.write(msg);
        g1.read_verify(msg);

        /* client destructor sends FI_SHUTDOWN to server */
        std::cerr << "CLIENT end " << iter0 << "." << iter1 << std::endl;
      }

      /* A special client to tell the server to shut down. Placed after other clients because the server apparently cannot abide concurrent clients. */
      remote_memory_client_for_shutdown client_shutdown(*fabric, "{}", remote_host, control_port);
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

    auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec_static));

    /* To avoid conflicts with server which are slow to shut down, use a different control port on every pass
     * But, what happens if a server is shut down (fi_shutdown) while a client is expecting to receive data?
     * Shouldn't the client see some sort of error?
     */
    auto control_port = std::uint16_t(control_port_2 + iter0);

    if ( is_server )
    {
      std::cerr << "SERVER begin " << iter0 << std::endl;
      {
        remote_memory_server_grouped server(*fabric, "{}", control_port);
        /* The server needs one permanent communicator, to handle the client
         * "disconnect" message.  * It does not need any other communicators,
         * but if it did, then the server (created by remote_memory_server_grouped
         * when it sees a client) would be the entity to create them.
         */
      }
      std::cerr << "SERVER end " << iter0 << std::endl;
    }
    else
    {
      /* allow time for the server to listen before the client restarts */
      std::this_thread::sleep_for(std::chrono::seconds(3));
      for ( auto iter1 = 0; iter1 != count_inner; ++iter1 )
      {
        std::cerr << "CLIENT begin " << iter0 << "." << iter1 << " port " << control_port << std::endl;
        remote_memory_client  client(*fabric, "{}", remote_host, control_port);

        std::string msg = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.";
        /* ought to split send from completions so that we can test the separation of comms */
        client.write(msg);
        client.read_verify(msg);

        /* client destructor sends FI_SHUTDOWN to server */
        std::cerr << "CLIENT end " << iter0 << "." << iter1 << std::endl;
      }

      /* A special client to tell the server to shut down. Placed after other clients because the server apparently cannot abide concurrent clients. */
      remote_memory_client_for_shutdown client_shutdown(*fabric, "{}", remote_host, control_port);
    }

    factory->release_ref();
  }
}

} // namespace

int main(int argc, char **argv)
{
  is_client = argv[1] && 0 == strcmp(argv[1], "client");
  is_server = argv[1] && 0 == strcmp(argv[1], "server");

  if ( ( ! is_client && ! is_server ) || ( is_client && argc < 3 ) )
  {
    PINF("%s [client <ipaddr> | server]", argv[0]);
    return -1;
  }
  remote_host = argv[2];

  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}

/* note: we do not include component source, only the API definition */
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

void instantiate_server(std::string fabric_spec_)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);

  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<Component::IFabric_factory *>(comp->query_interface(Component::IFabric_factory::iid())));
  auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec_));

  {
    auto srv1 = std::shared_ptr<Component::IFabric_server_factory>(fabric->open_server_factory("{}", control_port_0));
    describe_ep(std::cerr << "InstantiateServer " << fabric_spec_ << "Endpoint 1 ", *srv1) << std::endl;
  }
  {
    auto srv2 = std::shared_ptr<Component::IFabric_server_factory>(fabric->open_server_factory("{}", control_port_0));
    describe_ep(std::cerr << "InstantiateServer " << fabric_spec_ << " Endpoint 2 ", *srv2) << std::endl;
  }
  factory->release_ref();
}

static constexpr auto count_outer = 3U;
static constexpr auto count_inner = 3U;

void write_read_sequential(const std::string &fabric_spec_, bool force_error_)
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

    auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec_));
    if ( is_server )
    {
      std::cerr << "SERVER begin " << iter0 << " port " << control_port << std::endl;
      {
        auto remote_key_base = 0U;
        remote_memory_server server(*fabric, "{}", control_port, "", remote_key_base);
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
        remote_memory_client client(*fabric, "{}", remote_host, control_port, remote_key_index);

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
      remote_memory_client_for_shutdown client_shutdown(*fabric, "{}", remote_host, control_port, remote_key_index);
      EXPECT_EQ(client_shutdown.max_message_size(), msg_max);
    }

    factory->release_ref();
  }
}

TEST_F(Fabric_test, InstantiateServer)
{
  instantiate_server(fabric_spec("verbs"));
}

TEST_F(Fabric_test, InstantiateServer_Socket)
{
  instantiate_server(fabric_spec("sockets"));
}

void instantiate_server_dual(const std::string &fabric_spec_)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);

  ASSERT_TRUE(comp);
  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<Component::IFabric_factory *>(comp->query_interface(Component::IFabric_factory::iid())));
  auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec_));

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

TEST_F(Fabric_test, InstantiateServerDual)
{
  instantiate_server_dual(fabric_spec("verbs"));
}

TEST_F(Fabric_test, InstantiateServerDual_Sockets)
{
  instantiate_server_dual(fabric_spec("sockets"));
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

void instantiate_server_and_client(const std::string &fabric_spec_)
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
    auto server = std::shared_ptr<Component::IFabric_server_factory>(fabric0->open_server_factory(fabric_spec_, control_port_1));
    EXPECT_LT(0U, server->max_message_size());
    auto client = std::shared_ptr<Component::IFabric_client>(open_connection_patiently(*fabric1, fabric_spec_, "127.0.0.1", control_port_1));
    EXPECT_LT(0U, client->max_message_size());
    EXPECT_EQ(server->max_message_size(), client->max_message_size());
  }

  factory->release_ref();
}

TEST_F(Fabric_test, InstantiateServerAndClient)
{
  instantiate_server_and_client(fabric_spec("verbs"));
}

TEST_F(Fabric_test, InstantiateServerAndClientSockets)
{
  instantiate_server_and_client(fabric_spec("sockets"));
}

TEST_F(Fabric_test, WriteReadSequential)
{
  write_read_sequential(fabric_spec("verbs"), false);
}

TEST_F(Fabric_test, WriteReadSequentialSockets)
{
  write_read_sequential(fabric_spec("sockets"), false);
}

TEST_F(Fabric_test, WriteReadSequentialWithError)
{
  write_read_sequential(fabric_spec("sockets"), true);
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
        remote_memory_server server(*fabric, "{}", control_port, "", remote_key_base, expected_client_count);
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
          vv.emplace_back(*fabric, "{}", remote_host, control_port, remote_key_index);
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
        remote_memory_server server(*fabric, "{}", control_port, "", remote_key_base);
        EXPECT_LT(0U, server.max_message_size());
      }
      std::cerr << "SERVER end " << iter0 << std::endl;
    }
    else
    {
      /* allow time for the server to listen before the client restarts */
      std::this_thread::sleep_for(std::chrono::seconds(3));
      std::size_t msg_max(0U);
      for ( auto iter1 = 0; iter1 != count_inner; ++iter1 )
      {
        std::cerr << "CLIENT begin " << iter0 << "." << iter1 << " port " << control_port << std::endl;
        /* In case the provider actually uses the remote keys which we provide, make them unique. */
        auto remote_key_index = iter1 * 3U;

        remote_memory_client_grouped client(*fabric, "{}", remote_host, control_port, remote_key_index);

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
        remote_memory_subclient g0(client, remote_key_index + 1U);
        remote_memory_subclient g1(client, remote_key_index + 2U);

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
      remote_memory_client_for_shutdown client_shutdown(*fabric, "{}", remote_host, control_port, remote_key_index);
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
        remote_memory_server_grouped server(*fabric, "{}", control_port, remote_key_base);
        /* The server needs one permanent communicator, to handle the client
         * "disconnect" message.  * It does not need any other communicators,
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
        remote_memory_client  client(*fabric, "{}", remote_host, control_port, remote_key_index);

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
      remote_memory_client_for_shutdown client_shutdown(*fabric, "{}", remote_host, control_port, remote_key_index);
    }

    factory->release_ref();
  }
}

TEST_F(Fabric_test, PingPong)
{
  static constexpr size_t BUFFER_SIZE = 128;
  unsigned ITERATIONS = 10000;
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                    Component::net_fabric_factory);
  ASSERT_TRUE(comp);

  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<Component::IFabric_factory *>(comp->query_interface(Component::IFabric_factory::iid())));

  auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec("verbs")));

  auto control_port = std::uint16_t(control_port_2);

  auto start = std::chrono::high_resolution_clock::now();
  auto start_delay = std::chrono::seconds(3);

  if ( is_server )
  {
    std::cerr << "SERVER begin port " << control_port << std::endl;
    {
      auto remote_key_base = 0U;
      pingpong_server server(*fabric, "{}", control_port, remote_key_base, ITERATIONS, BUFFER_SIZE);
      EXPECT_LT(0U, server.max_message_size());
    }
    std::cerr << "SERVER end" << std::endl;
  }
  else
  {
    /* allow time for the server to listen before the client restarts */
    std::this_thread::sleep_for(start_delay);
    start = std::chrono::high_resolution_clock::now();
    /* In case the provider actually uses the remote keys which we provide, make them unique. */
    auto remote_key_base = 0U;

    std::cerr << "CLIENT begin port " << control_port << std::endl;
    pingpong_client  client(*fabric, "{}", remote_host, control_port, remote_key_base, ITERATIONS, BUFFER_SIZE);
    std::cerr << "CLIENT end" << std::endl;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto secs = double(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) / 1000.0;
  PINF("%zu byte PingPong iterations %u secs %f Ops/Sec: %lu", BUFFER_SIZE, ITERATIONS, secs, static_cast<unsigned long>( ITERATIONS / secs ));

  factory->release_ref();
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

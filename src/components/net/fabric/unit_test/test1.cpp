/* note: we do not include component source, only the API definition */
#include <gtest/gtest.h>


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <api/components.h>
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wvariadic-macros"
#include <api/fabric_itf.h>
#pragma GCC diagnostic pop

#include <sys/mman.h>
#include <sys/uio.h> /* iovec */
#include <cstring>
#include <memory>
#include <thread>

using namespace Component;

namespace {
  const std::string fabric_spec_static{
    "{ \"fabric_attr\" : { \"prov_name\" : \"verbs\" },"
    " \"domain_attr\" : "
      "{ \"mr_mode\" : [ \"FI_MR_LOCAL\", \"FI_MR_VIRT_ADDR\", \"FI_MR_ALLOCATED\", \"FI_MR_PROV_KEY\" ] }"
    ","
    " \"ep_attr\" : { \"type\" : \"FI_EP_MSG\" }"
    "}"
  }
;

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

std::ostream &describe_ep(std::ostream &o_, const IFabric_endpoint &e_)
{
  return o_ << "provider '" << e_.get_provider_name() << "' max_message_size " << e_.max_message_size();
}

namespace
{
  bool is_client = false;
  bool is_server = false;
  char *remote_host = nullptr;
  constexpr std::uint16_t control_port_0 = 47591;
  constexpr std::uint16_t control_port_1 = 47592;
  constexpr std::uint16_t control_port_2 = 47593;

  constexpr uint64_t remote_key = 54312U; /* value does not matter; ibverbs wil supply its own value */
  static constexpr std::size_t remote_memory_offset = 25;

  class registration
  {
    Component::IFabric_connection &_cnxn;
    Component::IFabric_connection::memory_region_t _region;
  public:
    explicit registration(Component::IFabric_connection &cnxn_, const void *contig_addr, size_t size, uint64_t key, uint64_t flags)
      : _cnxn(cnxn_)
      , _region(_cnxn.register_memory(contig_addr, size, key, flags))
    {
    }
    ~registration()
    {
      _cnxn.deregister_memory(_region);
    }
    std::uint64_t key() const { return std::get<1>(_region); }
  };

  class mr_lock
  {
    const void *_addr;
    std::size_t _len;
  public:
    mr_lock(const void *addr_, std::size_t len_)
      : _addr(addr_)
      , _len(len_)
    {
      if ( 0 != ::mlock(_addr, _len) )
      {
        auto e = errno;
        auto context = std::string("in ") + __func__;
        throw std::system_error{std::error_code{e, std::system_category()}, context};
      }
    }
    ~mr_lock()
    {
      ::munlock(_addr, _len);
    }
  };

  class registered_memory
  {
    Component::IFabric_connection &_cnxn;
    /* There may be an alignment restriction on registered memory, May be 8, or 16. */
    alignas(64) std::array<char, 4096> _memory;
    registration _registration;
  public:
    registered_memory(Component::IFabric_connection &cnxn_)
      : _cnxn(cnxn_)
      , _memory{}
      , _registration(_cnxn, &*_memory.begin(), _memory.size(), remote_key, 0U)
    {}

    char &operator[](std::size_t ix) { return _memory[ix]; }
    volatile char &first_char() { return _memory[0]; }

    std::uint64_t key() const { return _registration.key(); }
  };

  class remote_memory_server
  {
    std::shared_ptr<Component::IFabric_endpoint> _ep;
    std::thread _th;

    void wait_to_quit(registered_memory &rm, char rma_first_char)
    {
      /* serve as a remote buffer until the client writes 'q' to the magic location. */
      auto volatile quit_byte = &rm[remote_memory_offset];
      while ( *quit_byte != 'q' )
      {
        auto rma_new_first_char = rm.first_char();
        if ( rma_new_first_char != rma_first_char )
        {
          std::cerr << "Server buffer char changes from " << unsigned(uint8_t(rma_first_char)) << " to " << unsigned(uint8_t(rma_new_first_char)) << "\n";
          rma_first_char = rma_new_first_char;
        }
      }
std::cerr << "Server received 'q'" << std::endl;
    }

    void send_memory_info(Component::IFabric_connection &cnxn, registered_memory &rm)
    {
      /* get the virtual address of the buffer and send it to the client */
      std::vector<iovec> v;
      std::uint64_t vaddr = reinterpret_cast<std::uint64_t>(&rm[0]);
      std::uint64_t key = rm.key();
      std::cerr << "Server vaddr " << vaddr << " key " << key << std::endl;
      /* using _memory as the buffer (since it is already registered),
       * send memory's virtual address to the client
       */
      std::memcpy(&rm[0], &vaddr, sizeof vaddr);
      std::memcpy(&rm[sizeof vaddr], &key, sizeof key);
      iovec iv;
      iv.iov_base = &rm[0];
      iv.iov_len = (sizeof vaddr) + (sizeof key);
      v.emplace_back(iv);
      cnxn.post_send(v, this);
      {
        auto completed_ct = 0;
        while ( completed_ct == 0 )
        {
          auto ct =
            cnxn.poll_completions(
              [&v,this] (void *ctxt, status_t st) -> void
              {
                assert(ctxt == this);
                assert(st == S_OK);
              }
            );
          if ( ct != 0 ) {
          }
          completed_ct += ct;
        }
      }
    }

    void listener(Component::IFabric_endpoint &ep)
    {
      Component::IFabric_connection *cnxn = nullptr;
      /* busy-wait for a client */
      while ( ! ( cnxn = ep.get_new_connections() ) ) {}
      /* register an RDMA memory region */
      registered_memory rm{*cnxn};
      auto rma_first_char = rm.first_char();
      /* send the client address and key to memory */
      send_memory_info(*cnxn, rm);
      /* Wait for the client to signal "quit" */
      wait_to_quit(rm, rma_first_char);
      ep.close_connection(cnxn);
    }
  public:
    remote_memory_server(Component::IFabric &fabric_, const std::string &fabric_spec_, std::uint16_t control_port_)
      : _ep(fabric_.open_endpoint(fabric_spec_, control_port_))
      , _th(&remote_memory_server::listener, this, std::ref(*_ep))
    {
    }
    ~remote_memory_server()
    {
      _th.join();
    }
  };

  class remote_memory_client
  {
    static void check_complete_static(void *rmc_, status_t stat_)
    {
      auto rmc = static_cast<remote_memory_client *>(rmc_);
      ASSERT_TRUE(rmc);
      rmc->check_complete(stat_);
    }
    void check_complete(status_t stat_)
    {
      ASSERT_TRUE(stat_ == S_OK);
    }
    std::shared_ptr<Component::IFabric_connection> _cnxn;
    registered_memory rm_out;
    registered_memory rm_in;
    std::uint64_t _vaddr;
    std::uint64_t _key;
  public:
    remote_memory_client(Component::IFabric &fabric_, const std::string &fabric_spec_, const std::string ip_address_, std::uint16_t port_)
      : _cnxn(fabric_.open_connection(fabric_spec_, ip_address_, port_))
      , rm_out{*_cnxn}
      , rm_in{*_cnxn}
      , _vaddr{}
      , _key{}
    {
      std::vector<iovec> v;
      iovec iv;
      iv.iov_base = &rm_out[0];
      iv.iov_len = (sizeof _vaddr) + (sizeof _key);
      v.emplace_back(iv);
      _cnxn->post_recv(v, this);
      {
        auto completed_ct = 0;
        while ( completed_ct == 0 )
        {
          auto ct =
            _cnxn->poll_completions(
              [&v, this] (void *ctxt, status_t st) -> void
              {
                assert(ctxt == this);
                assert(st == S_OK);
                assert(v[0].iov_len == (sizeof _vaddr) + sizeof( _key));
                std::memcpy(&_vaddr, &rm_out[0], sizeof _vaddr);
                std::memcpy(&_key, &rm_out[sizeof _vaddr], sizeof _key);
              }
            );
          completed_ct += ct;
        }
      }
      std::cerr << "Remote memory client addr " << _vaddr << " key " << _key << std::endl;
    }
    void write(const std::string &msg)
    {
      std::copy(msg.begin(), msg.end(), &rm_out[0]);
      std::vector<iovec> buffers(1);
      {
std::cerr << "Posting write from " << static_cast<void *>(&rm_out[0]) << " to " << _vaddr + remote_memory_offset << " key " << _key << std::endl;
        buffers[0].iov_base = &rm_out[0];
        buffers[0].iov_len = msg.size();
        _cnxn->post_write(buffers, _vaddr + remote_memory_offset, _key, this);
      }
      {
#if 0
/* We do not yet support wait_for_next_completion */
        _cnxn->wait_for_next_completion();
        auto ct = _cnxn->poll_completions(check_complete_static);
        ASSERT_TRUE(ct == 1);
#else
        while ( ! _cnxn->poll_completions(check_complete_static) ) {}
#endif
      }
    }
    void read_verify(const std::string &msg)
    {
      std::vector<iovec> buffers(1);
      {
std::cerr << "Posting read to " << static_cast<void *>(&rm_in[0]) << " from " << _vaddr + remote_memory_offset << std::endl;
        buffers[0].iov_base = &rm_in[0];
        buffers[0].iov_len = msg.size();
        _cnxn->post_read(buffers, _vaddr + remote_memory_offset, _key, this);
      }
      {
#if 0
/* We do not yet support wait_for_next_completion */
        _cnxn->wait_for_next_completion();
        auto ct = _cnxn->poll_completions(check_complete_static);
        ASSERT_TRUE(ct == 1);
#else
        while ( ! _cnxn->poll_completions(check_complete_static) ) {}
#endif
      }
      std::string remote_msg(&rm_in[0], &rm_in[0] + msg.size());
      ASSERT_TRUE(msg == remote_msg);
std::cerr << "Client verified readback" << std::endl;
    }
  };
}

TEST_F(Fabric_test, InstantiateServer)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);

  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));
  auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec));

  {
    auto srv1 = std::shared_ptr<Component::IFabric_endpoint>(fabric->open_endpoint("{}", control_port_1));
    describe_ep(std::cerr << "IS Endpoint 1 ", *srv1) << "\n";
  }
  {
    auto srv2 = std::shared_ptr<Component::IFabric_endpoint>(fabric->open_endpoint("{}", control_port_1));
    describe_ep(std::cerr << "IS Endpoint 2 ", *srv2) << "\n";
  }
  factory->release_ref();
}

TEST_F(Fabric_test, InstantiateServerDual)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);

  ASSERT_TRUE(comp);
  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));
  auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec));

  try {
    /* fails, because both servers use the same port */
    auto srv1 = std::shared_ptr<Component::IFabric_endpoint>(fabric->open_endpoint("{}", control_port_1));
    auto srv2 = std::shared_ptr<Component::IFabric_endpoint>(fabric->open_endpoint("{}", control_port_2));
    describe_ep(std::cerr << "ISD Endpoint 1 ", *srv1) << "\n";
    describe_ep(std::cerr << "ISD Endpoint 2 ", *srv2) << "\n";
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
  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));
  auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec));
  /* Feed the endpoint a good JSON spec */

  try
  {
    auto pep1 = std::shared_ptr<Component::IFabric_endpoint>(fabric->open_endpoint("{ \"tx_attr\" : { \"comp_order\" : [ \"FI_ORDER_STRICT\", \"FI_ORDER_DATA\" ], \"inject_size\" : 16 } }", control_port_1));
    describe_ep(std::cerr << "Endpoint 1 ", *pep1) << "\n";
  }
  catch ( const std::domain_error &e )
  {
    std::cerr << "Unexpected exception: " << e.what() << "\n";
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
  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));
  auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec));

  {
    /* Feed the ednpoint a good JSON spec */
    try
    {
      auto pep1 = std::shared_ptr<Component::IFabric_endpoint>(fabric->open_endpoint(("{ \"addr_format\" : \"FI_ADDR_STR\", \"dest\" : \"fi_shm://" + std::to_string(getpid()) + "\" }").c_str(), control_port_1));
    }
    catch ( const std::exception &e )
    {
      std::cerr << "Unexpected exception: " << e.what() << "\n";
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
  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));
  auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec));

  {
    /* Feed the ednpoint a good JSON spec */
    try
    {
      auto pep1 = std::shared_ptr<Component::IFabric_endpoint>(fabric->open_endpoint("{ \"tx_attr\" : { \"max\" : \"xyz\"} } }", control_port_1));
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
  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));
  auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec));

  {
    /* Feed the ednpoint a good JSON spec */
    try
    {
      auto pep1 = std::shared_ptr<Component::IFabric_endpoint>(fabric->open_endpoint("{ \"tx_attr\" : { \"maX\" : \"xyz\"} }", control_port_1));
      ASSERT_TRUE(false);
    }
    catch ( const std::domain_error &e )
    {
      /* Error mesage should mention "key", "tx_attr", and "maX" */
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
  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));
  auto fabric0 = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec));
  auto fabric1 = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec));

  {
    /* Feed the endpoint a good JSON spec */
    auto server = std::shared_ptr<Component::IFabric_endpoint>(fabric0->open_endpoint(fabric_spec, control_port_1));
    auto client = std::shared_ptr<Component::IFabric_connection>(fabric1->open_connection(fabric_spec, "127.0.0.1", control_port_1));
  }

  factory->release_ref();
}

TEST_F(Fabric_test, WriteRead)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);
  ASSERT_TRUE(comp);

  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));

  if ( is_client )
  {
    auto fabric_client = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec_static));
    /* Feed the endpoint a good JSON spec */
    remote_memory_client client(*fabric_client, "{}", remote_host, control_port_0);
    /* Feed the endpoint some terrible text */
    std::string msg = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.";
    client.write(msg);
    client.read_verify(msg);
    client.write("q");
  }
  else
  {
    auto fabric_server = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec_static));
    remote_memory_server server(*fabric_server, "{}", control_port_0);
  }

  factory->release_ref();
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

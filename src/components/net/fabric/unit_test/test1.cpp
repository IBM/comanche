/* note: we do not include component source, only the API definition */
#include <gtest/gtest.h>


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#include <api/components.h>
#pragma GCC diagnostic pop

#include <api/fabric_itf.h>

#include <sys/mman.h>
#include <sys/uio.h> /* iovec */
#include <chrono> /* seconds */
#include <cstring>
#include <memory>
#include <thread> /* sleep_for */

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
  const char *eyecatcher = "=====================================";

  constexpr std::size_t remote_memory_offset = 25;

  Component::IFabric_connection *open_connection_patiently(Component::IFabric &fabric_, const std::string &fabric_spec_, const std::string ip_address_, std::uint16_t port_)
  {
    Component::IFabric_connection *cnxn = nullptr;
    int try_count = 0;
    while ( ! cnxn )
    {
      try
      {
        cnxn = fabric_.open_connection(fabric_spec_, ip_address_, port_);
      }
      catch ( std::system_error &e )
      {
        if ( e.code().value() != ECONNREFUSED )
        {
          throw;
        }
      }
      ++try_count;
    }
    return cnxn;
  }

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
    try
    {
      _cnxn.deregister_memory(_region);
    }
    catch ( std::exception &e )
    {
      std::cerr << __func__ << " exception " << e.what() << eyecatcher << std::endl;
    }

    std::uint64_t key() const { return std::get<1>(_region); }
  };

  class mr_lock
  {
    const void *_addr;
    std::size_t _len;
    mr_lock(const mr_lock &) = delete;
    mr_lock& operator=(const mr_lock &) = delete;
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
    try
    {
      ::munlock(_addr, _len);
    }
    catch ( std::exception &e )
    {
      std::cerr << __func__ << " exception " << e.what() << eyecatcher << std::endl;
    }
  };

  class server_connection
  {
    Component::IFabric_endpoint &_ep;
    Component::IFabric_connection *_cnxn;
  public:
    Component::IFabric_connection &cnxn() const { return *_cnxn; }
    server_connection(Component::IFabric_endpoint &ep_)
      : _ep(ep_)
      , _cnxn(nullptr)
    {
      while ( ! ( _cnxn = _ep.get_new_connections() ) ) {}
    }
    server_connection(const server_connection &) = delete;
    server_connection& operator=(const server_connection &) = delete;
    ~server_connection()
    try
    {
      _ep.close_connection(_cnxn);
    }
    catch ( std::exception &e )
    {
      std::cerr << __func__ << " exception " << e.what() << eyecatcher << std::endl;
    }
  };

  class registered_memory
  {
    static constexpr uint64_t remote_key = 54312U; /* value does not matter to ibverbs; ibverbs ignores the value and creates its own key */
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

  void wait_poll(Component::IFabric_communicator &comm, std::function<void(void *context, status_t)> cb)
  {
    size_t ct = 0;
    unsigned delay = 0;
    while ( ct == 0 )
    {
      comm.wait_for_next_completion(std::chrono::seconds(6000));
      ct = comm.poll_completions(cb);
      ++delay;
    }
    /* poll_completions does not always get a completion after wait_for_next_completion returns
     * (does it perhaps return when a message begins to appear in the completion queue?)
     * but it should not take more than two trips through the loop to get the completion.
     */
    ASSERT_LE(delay,2);
    ASSERT_EQ(ct,1);
  }

  class remote_memory_accessor
  {
  public:
    /* using rm as a buffer, send message */
    void send_msg(Component::IFabric_connection &cnxn, registered_memory &rm, const void *msg, std::size_t len)
    {
      std::vector<iovec> v;
      std::memcpy(&rm[0], msg, len);
      iovec iv;
      iv.iov_base = &rm[0];
      iv.iov_len = len;
      v.emplace_back(iv);
      try
      {
        cnxn.post_send(v, this);
        wait_poll(
          cnxn
          , [&v,this] (void *ctxt, status_t st) -> void
            {
              ASSERT_EQ(ctxt, this);
              ASSERT_EQ(st, S_OK);
            }
        );
      }
      catch ( const std::exception &e )
      {
        std::cerr << __func__ << " exception " << e.what() << eyecatcher << std::endl;
      }
    }
  };

  /*
   * A Component::IFabric_endpoint ought to be able to support multiple clients,
   * but remote_memory_server uses it for jsut a single client, then folds up shop.
   */
  class remote_memory_server
    : public remote_memory_accessor
  {
    std::shared_ptr<Component::IFabric_endpoint> _ep;
    std::thread _th;

    void send_memory_info(Component::IFabric_connection &cnxn_, registered_memory &rm_)
    {
      std::uint64_t vaddr = reinterpret_cast<std::uint64_t>(&rm_[0]);
      std::uint64_t key = rm_.key();
      char msg[(sizeof vaddr) + (sizeof key)];
      std::memcpy(msg, &vaddr, sizeof vaddr);
      std::memcpy(&msg[sizeof vaddr], &key, sizeof key);
      send_msg(cnxn_, rm_, msg, sizeof msg);
    }

    void listener(Component::IFabric_endpoint &ep_)
    {
      auto quit = false;
      while ( ! quit )
      {
        server_connection sc(ep_);
        /* register an RDMA memory region */
        registered_memory rm{sc.cnxn()};
        /* send the client address and key to memory */
        send_memory_info(sc.cnxn(), rm);
        /* wait for client indicate exit (by sending one byte to us */
        {
          std::vector<iovec> v;
          iovec iv;
          iv.iov_base = &rm[0];
          iv.iov_len = 1;
          v.emplace_back(iv);
          sc.cnxn().post_recv(v, this);
          wait_poll(
            sc.cnxn()
            , [&v, &quit, &rm, this] (void *ctxt, status_t st) -> void
              {
                ASSERT_EQ(ctxt, this);
                ASSERT_EQ(st, S_OK);
                ASSERT_EQ(v[0].iov_len, 1);
                /* did client leave with the "quit byte" set to 'q'? */
                quit |= rm[0] == 'q';
              }
          );
        }
      }
    }
  public:
    remote_memory_server(Component::IFabric &fabric_, const std::string &fabric_spec_, std::uint16_t control_port_)
      : _ep(fabric_.open_endpoint(fabric_spec_, control_port_))
      , _th(&remote_memory_server::listener, this, std::ref(*_ep))
    {
    }
    ~remote_memory_server()
    try
    {
      _th.join();
    }
    catch ( std::exception &e )
    {
      std::cerr << __func__ << " exception " << e.what() << eyecatcher << std::endl;
    }
  };

  class remote_memory_client
    : public remote_memory_accessor
  {
    static void check_complete_static(void *rmc_, status_t stat_)
    {
      auto rmc = static_cast<remote_memory_client *>(rmc_);
      ASSERT_TRUE(rmc);
      rmc->check_complete(stat_);
    }

    static void check_complete_static_2(void *t_, void *rmc_, status_t stat_)
    {
      /* The callback context must be the object which was polling. */
      ASSERT_EQ(t_, rmc_);
      check_complete_static(rmc_, stat_);
    }

    static void check_complete(status_t stat_)
    {
      ASSERT_EQ(stat_, S_OK);
    }

    std::shared_ptr<Component::IFabric_connection> _cnxn;
    registered_memory _rm_out;
    registered_memory _rm_in;
    std::uint64_t _vaddr;
    std::uint64_t _key;
    char _quit_flag;
  protected:
    void do_quit()
    {
      _quit_flag = 'q';
    }
  public:
    remote_memory_client(Component::IFabric &fabric_, const std::string &fabric_spec_, const std::string ip_address_, std::uint16_t port_)
      : _cnxn(open_connection_patiently(fabric_, fabric_spec_, ip_address_, port_))
      , _rm_out{*_cnxn}
      , _rm_in{*_cnxn}
      , _vaddr{}
      , _key{}
      , _quit_flag('n')
    {
      std::vector<iovec> v;
      iovec iv;
      iv.iov_base = &_rm_out[0];
      iv.iov_len = (sizeof _vaddr) + (sizeof _key);
      v.emplace_back(iv);
      _cnxn->post_recv(v, this);
      wait_poll(
        *_cnxn
        , [&v, this] (void *ctxt, status_t st) -> void
          {
            ASSERT_EQ(ctxt, this);
            ASSERT_EQ(st, S_OK);
            ASSERT_EQ(v[0].iov_len, (sizeof _vaddr) + sizeof( _key));
            std::memcpy(&_vaddr, &_rm_out[0], sizeof _vaddr);
            std::memcpy(&_key, &_rm_out[sizeof _vaddr], sizeof _key);
          }
      );
      std::cerr << "Remote memory client addr " << _vaddr << " key " << _key << std::endl;
    }

    void send_disconnect(Component::IFabric_connection &cnxn, registered_memory &rm, char quit_flag)
    {
      send_msg(cnxn, rm, &quit_flag, sizeof quit_flag);
    }

    ~remote_memory_client()
    try
    {
      send_disconnect(cnxn(), _rm_out, _quit_flag);
    }
    catch ( std::exception &e )
    {
      std::cerr << __func__ << " exception " << e.what() << eyecatcher << std::endl;
    }

    std::uint64_t vaddr() const { return _vaddr; }
    std::uint64_t key() const { return _key; }
    Component::IFabric_connection &cnxn() { return *_cnxn; }

    void write(const std::string &msg)
    {
      std::copy(msg.begin(), msg.end(), &_rm_out[0]);
      std::vector<iovec> buffers(1);
      {
        buffers[0].iov_base = &_rm_out[0];
        buffers[0].iov_len = msg.size();
        _cnxn->post_write(buffers, _vaddr + remote_memory_offset, _key, this);
      }
      wait_poll(
        *_cnxn
        , [this] (void *rmc_, status_t stat_) { check_complete_static_2(this, rmc_, stat_); } /* WAS: check_complete_static */
      );

    }

    void read_verify(const std::string &msg)
    {
      std::vector<iovec> buffers(1);
      {
        buffers[0].iov_base = &_rm_in[0];
        buffers[0].iov_len = msg.size();
        _cnxn->post_read(buffers, _vaddr + remote_memory_offset, _key, this);
      }
      wait_poll(
        *_cnxn
        , [this] (void *rmc_, status_t stat_) { check_complete_static_2(this, rmc_, stat_); } /* WAS: check_complete_static */
      );
      std::string remote_msg(&_rm_in[0], &_rm_in[0] + msg.size());
      ASSERT_EQ(msg, remote_msg);
    }

    Component::IFabric_communicator *allocate_group() const
    {
      return _cnxn->allocate_group();
    }
  };

  class remote_memory_client_for_shutdown
    : private remote_memory_client
  {
  public:
    remote_memory_client_for_shutdown(Component::IFabric &fabric_, const std::string &fabric_spec_, const std::string ip_address_, std::uint16_t port_)
      : remote_memory_client(fabric_, fabric_spec_, ip_address_, port_)
    {
      do_quit();
    }
  };

  class remote_memory_subclient
  {
    static void check_complete_static(void *rmc_, status_t stat_)
    {
      auto rmc = static_cast<remote_memory_subclient *>(rmc_);
      ASSERT_TRUE(rmc);
      rmc->check_complete(stat_);
    }
    static void check_complete_static_2(void *t_, void *rmc_, status_t stat_)
    {
      /* The callback context must be the object which was polling. */
      ASSERT_EQ(t_, rmc_);
      check_complete_static(rmc_, stat_);
    }
    void check_complete(status_t stat_)
    {
      ASSERT_EQ(stat_, S_OK);
    }
    remote_memory_client &_parent;
    std::shared_ptr<Component::IFabric_communicator> _cnxn;
    registered_memory _rm_out;
    registered_memory _rm_in;
  public:
    remote_memory_subclient(remote_memory_client &parent_)
      : _parent(parent_)
      , _cnxn(_parent.allocate_group())
      , _rm_out{_parent.cnxn()}
      , _rm_in{_parent.cnxn()}
    {
    }

    void write(const std::string &msg)
    {
      std::copy(msg.begin(), msg.end(), &_rm_out[0]);
      std::vector<iovec> buffers(1);
      {
        buffers[0].iov_base = &_rm_out[0];
        buffers[0].iov_len = msg.size();
        _cnxn->post_write(buffers, _parent.vaddr() + remote_memory_offset, _parent.key(), this);
      }
      wait_poll(
        *_cnxn
        , [this] (void *rmc_, status_t stat_) { check_complete_static_2(this, rmc_, stat_); } /* WAS: check_complete_static */
      );
    }

    void read_verify(const std::string &msg)
    {
      std::vector<iovec> buffers(1);
      {
        buffers[0].iov_base = &_rm_in[0];
        buffers[0].iov_len = msg.size();
        _cnxn->post_read(buffers, _parent.vaddr() + remote_memory_offset, _parent.key(), this);
      }
      wait_poll(
        *_cnxn
        , [this] (void *rmc_, status_t stat_) { check_complete_static_2(this, rmc_, stat_); } /* WAS: check_complete_static */
      );
      std::string remote_msg(&_rm_in[0], &_rm_in[0] + msg.size());
      ASSERT_EQ(msg, remote_msg);
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
    auto srv1 = std::shared_ptr<Component::IFabric_endpoint>(fabric->open_endpoint("{}", control_port_0));
    describe_ep(std::cerr << "IS Endpoint 1 ", *srv1) << std::endl;
  }
  {
    auto srv2 = std::shared_ptr<Component::IFabric_endpoint>(fabric->open_endpoint("{}", control_port_0));
    describe_ep(std::cerr << "IS Endpoint 2 ", *srv2) << std::endl;
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
    auto srv1 = std::shared_ptr<Component::IFabric_endpoint>(fabric->open_endpoint("{}", control_port_0));
    auto srv2 = std::shared_ptr<Component::IFabric_endpoint>(fabric->open_endpoint("{}", control_port_1));
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
  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));
  auto fabric = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec));
  /* Feed the endpoint a good JSON spec */

  try
  {
    auto pep1 = std::shared_ptr<Component::IFabric_endpoint>(fabric->open_endpoint("{ \"tx_attr\" : { \"comp_order\" : [ \"FI_ORDER_STRICT\", \"FI_ORDER_DATA\" ], \"inject_size\" : 16 } }", control_port_1));
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
  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));
  auto fabric0 = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec));
  auto fabric1 = std::shared_ptr<Component::IFabric>(factory->make_fabric(fabric_spec));

  {
    /* Feed the endpoint a good JSON spec */
    auto server = std::shared_ptr<Component::IFabric_endpoint>(fabric0->open_endpoint(fabric_spec, control_port_1));
    auto client = std::shared_ptr<Component::IFabric_connection>(open_connection_patiently(*fabric1, fabric_spec, "127.0.0.1", control_port_1));
  }

  factory->release_ref();
}

static constexpr auto count_outer = 5U;
static constexpr auto count_inner = 10U;

TEST_F(Fabric_test, WriteRead)
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

    auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));

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
        /* Feed the endpoint a good JSON spec */
        remote_memory_client client(*fabric, "{}", remote_host, control_port);
        /* Feed the endpoint some terrible text */
        std::string msg = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.";
        /* An ordinary client which tests RDMA to server memory */
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

TEST_F(Fabric_test, Groups)
{
  for ( uint16_t iter0 = 0U; iter0 != count_outer; ++iter0 )
  {
    /* create object instance through factory */
    Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);
    ASSERT_TRUE(comp);

    auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));

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

        /* Feed the endpoint a good JSON spec */
        remote_memory_client client(*fabric, "{}", remote_host, control_port);

        /* make one "communicator (two, including the parent. */
        remote_memory_subclient g0(client);
        std::string msg = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.";
        /* ought to split send from completions so that we can test the separation of comms */
        g0.write(msg);
        g0.read_verify(msg);
        /* client destructor sends FI_SHUTDOWN to server */
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

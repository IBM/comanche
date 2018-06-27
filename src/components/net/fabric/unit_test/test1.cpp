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
#include <boost/core/noncopyable.hpp>
#include <boost/io/ios_state.hpp>
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

std::ostream &describe_ep(std::ostream &o_, const IFabric_server_factory &e_)
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

  Component::IFabric_client *open_connection_patiently(Component::IFabric &fabric_, const std::string &fabric_spec_, const std::string ip_address_, std::uint16_t port_)
  {
    Component::IFabric_client *cnxn = nullptr;
    int try_count = 0;
    while ( ! cnxn )
    {
      try
      {
        cnxn = fabric_.open_client(fabric_spec_, ip_address_, port_);
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

  Component::IFabric_client_grouped *open_connection_grouped_patiently(Component::IFabric &fabric_, const std::string &fabric_spec_, const std::string ip_address_, std::uint16_t port_)
  {
    Component::IFabric_client_grouped *cnxn = nullptr;
    int try_count = 0;
    while ( ! cnxn )
    {
      try
      {
        cnxn = fabric_.open_client_grouped(fabric_spec_, ip_address_, port_);
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
    : private boost::noncopyable
  {
    Component::IFabric_connection &_cnxn;
    Component::IFabric_connection::memory_region_t _region;
  public:
    explicit registration(Component::IFabric_connection &cnxn_, const void *contig_addr_, size_t size_, uint64_t key_, uint64_t flags_)
      : _cnxn(cnxn_)
      , _region(_cnxn.register_memory(contig_addr_, size_, key_, flags_))
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
    Component::IFabric_server_factory &_ep;
    Component::IFabric_server *_cnxn;
    server_connection(const server_connection &) = delete;
    server_connection& operator=(const server_connection &) = delete;
  public:
    Component::IFabric_server &cnxn() const { return *_cnxn; }
    server_connection(Component::IFabric_server_factory &ep_)
      : _ep(ep_)
      , _cnxn(nullptr)
    {
      while ( ! ( _cnxn = _ep.get_new_connections() ) ) {}
    }
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

  class server_grouped_connection
  {
    Component::IFabric_server_grouped_factory &_ep;
    Component::IFabric_server_grouped *_cnxn;
    Component::IFabric_communicator *_comm;

    server_grouped_connection(const server_grouped_connection &) = delete;
    server_grouped_connection& operator=(const server_grouped_connection &) = delete;
    static Component::IFabric_server_grouped *get_connection(Component::IFabric_server_grouped_factory &ep_)
    {
      Component::IFabric_server_grouped *cnxn = nullptr;
      while ( ! ( cnxn = ep_.get_new_connections() ) ) {}
      return cnxn;
    }
  public:
    Component::IFabric_server_grouped &cnxn() const { return *_cnxn; }
    server_grouped_connection(Component::IFabric_server_grouped_factory &ep_)
      : _ep(ep_)
      , _cnxn(get_connection(_ep))
      , _comm(_cnxn->allocate_group())
    {
    }
    ~server_grouped_connection()
    try
    {
      delete _comm;
      _ep.close_connection(_cnxn);
    }
    catch ( std::exception &e )
    {
      std::cerr << __func__ << " exception " << e.what() << eyecatcher << std::endl;
    }
    Component::IFabric_communicator &comm() const { return *_comm; }
    Component::IFabric_communicator *allocate_group() const { return cnxn().allocate_group(); }
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

  void wait_poll(Component::IFabric_communicator &comm_, std::function<void(void *context, status_t)> cb_)
  {
    size_t ct = 0;
    unsigned delay = 0;
    while ( ct == 0 )
    {
      comm_.wait_for_next_completion(std::chrono::seconds(6000));
      ct = comm_.poll_completions(cb_);
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
  protected:
    void send_memory_info(Component::IFabric_communicator &cnxn_, registered_memory &rm_)
    {
      std::uint64_t vaddr = reinterpret_cast<std::uint64_t>(&rm_[0]);
      std::uint64_t key = rm_.key();
      char msg[(sizeof vaddr) + (sizeof key)];
      std::memcpy(msg, &vaddr, sizeof vaddr);
      std::memcpy(&msg[sizeof vaddr], &key, sizeof key);
      send_msg(cnxn_, rm_, msg, sizeof msg);
    }
  public:
    /* using rm as a buffer, send message */
    void send_msg(Component::IFabric_communicator &cnxn_, registered_memory &rm_, const void *msg_, std::size_t len_)
    {
      std::vector<iovec> v;
      std::memcpy(&rm_[0], msg_, len_);
      iovec iv;
      iv.iov_base = &rm_[0];
      iv.iov_len = len_;
      v.emplace_back(iv);
      try
      {
        cnxn_.post_send(v, this);
        wait_poll(
          cnxn_
          , [&v, this] (void *ctxt, status_t st) -> void
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

  class server_connection_and_memory
    : public server_connection
    , public registered_memory
    , public remote_memory_accessor
    , private boost::noncopyable
  {
  public:
    server_connection_and_memory(Component::IFabric_server_factory &ep_)
      : server_connection(ep_)
      , registered_memory(cnxn())
    {
      /* send the address, and the key to memory */
      send_memory_info(cnxn(), *this);
    }
    ~server_connection_and_memory()
    {
      std::vector<iovec> v;
      iovec iv;
      iv.iov_base = &((*this)[0]);
      iv.iov_len = 1;
      v.emplace_back(iv);
      cnxn().post_recv(v, this);
      wait_poll(
        cnxn()
        , [&v, this] (void *ctxt, status_t st) -> void
          {
            ASSERT_EQ(ctxt, this);
            ASSERT_EQ(st, S_OK);
            ASSERT_EQ(v[0].iov_len, 1);
          }
      );
    }
  };

  /*
   * A Component::IFabric_server_factory, which will support clients until one
   * of them closes with the "quit" flag set.
   */
  class remote_memory_server
    : public remote_memory_accessor
    , private boost::noncopyable
  {
    std::shared_ptr<Component::IFabric_server_factory> _ep;
    std::thread _th;

    void listener(Component::IFabric_server_factory &ep_)
    {
      auto quit = false;
      while ( ! quit )
      {
        server_connection sc(ep_);
        /* register an RDMA memory region */
        registered_memory rm{sc.cnxn()};
        /* send the client address and key to memory */
        send_memory_info(sc.cnxn(), rm);
        /* wait for client indicate exit (by sending one byte to us) */
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

    void listener_counted(Component::IFabric_server_factory &ep_, unsigned cnxn_count_)
    {
      std::vector<std::shared_ptr<server_connection_and_memory>> scrm;
      for ( auto i = 0U; i != cnxn_count_; ++i )
      {
        scrm.emplace_back(std::make_shared<server_connection_and_memory>(ep_));
      }
    }
  public:
    remote_memory_server(Component::IFabric &fabric_, const std::string &fabric_spec_, std::uint16_t control_port_)
      : _ep(fabric_.open_server_factory(fabric_spec_, control_port_))
      , _th(&remote_memory_server::listener, this, std::ref(*_ep))
    {
    }
    remote_memory_server(Component::IFabric &fabric_, const std::string &fabric_spec_, std::uint16_t control_port_, unsigned cnxn_limit_)
      : _ep(fabric_.open_server_factory(fabric_spec_, control_port_))
      , _th(&remote_memory_server::listener_counted, this, std::ref(*_ep), cnxn_limit_)
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

  class remote_memory_server_grouped;

  class remote_memory_subserver
  {
    static void check_complete_static(void *rmc_, status_t stat_)
    {
      auto rmc = static_cast<remote_memory_subserver *>(rmc_);
      ASSERT_TRUE(rmc);
      rmc->check_complete(stat_);
    }
    static void check_complete_static_2(void *t_, void *rmc_, status_t stat_)
    {
      /* The callback context must be the object which was polling. */
      ASSERT_EQ(t_, rmc_);
      check_complete_static(rmc_, stat_);
    }
public:
    void check_complete(status_t stat_)
    {
      ASSERT_EQ(stat_, S_OK);
    }
private:
    server_grouped_connection &_parent;
    std::shared_ptr<Component::IFabric_communicator> _cnxn;
    registered_memory _rm_out;
    registered_memory _rm_in;
    registered_memory &rm_out() { return _rm_out; }
    registered_memory &rm_in () { return _rm_in; }
  public:
    remote_memory_subserver(server_grouped_connection &parent_);

    Component::IFabric_communicator &cnxn() { return *_cnxn; }
  };

  class remote_memory_server_grouped
    : public remote_memory_accessor
    , private boost::noncopyable
  {
    std::shared_ptr<Component::IFabric_server_grouped_factory> _ep;
    std::thread _th;

    void listener(Component::IFabric_server_grouped_factory &ep_)
    {
      auto quit = false;
      while ( ! quit )
      {
        /* Get a client to work with */
        /* Get a client to work with */
        server_grouped_connection sc(ep_);
        /* register an RDMA memory region */
        registered_memory rm{sc.cnxn()};
        /* send the client address and key to memory */
        auto &cnxn = sc.comm();
        send_memory_info(cnxn, rm);
        /* wait for client indicate exit (by sending one byte to us) */
        {
          std::vector<iovec> v;
          iovec iv;
          iv.iov_base = &rm[0];
          iv.iov_len = 1;
          v.emplace_back(iv);
          cnxn.post_recv(v, this);
          wait_poll(
            cnxn
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
    remote_memory_server_grouped(Component::IFabric &fabric_, const std::string &fabric_spec_, std::uint16_t control_port_)
      : _ep(fabric_.open_server_grouped_factory(fabric_spec_, control_port_))
      , _th(&remote_memory_server_grouped::listener, this, std::ref(*_ep))
    {}

    ~remote_memory_server_grouped()
    try
    {
      _th.join();
    }
    catch ( std::exception &e )
    {
      std::cerr << __func__ << " exception " << e.what() << eyecatcher << std::endl;
    }
  };

  remote_memory_subserver::remote_memory_subserver(server_grouped_connection &parent_)
    : _parent(parent_)
    , _cnxn(_parent.allocate_group())
    , _rm_out{_parent.cnxn()}
    , _rm_in{_parent.cnxn()}
  {
  }

  class remote_memory_client_grouped
    : public remote_memory_accessor
  {
    static void check_complete_static(void *rmc_, status_t stat_);

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

    std::shared_ptr<Component::IFabric_client_grouped> _cnxn;
    std::shared_ptr<registered_memory> _rm_out;
    std::shared_ptr<registered_memory> _rm_in;
    std::uint64_t _vaddr;
    std::uint64_t _key;
    char _quit_flag;

    registered_memory &rm_in() const { return *_rm_in; }
    registered_memory &rm_out() const { return *_rm_out; }

  public:
    remote_memory_client_grouped(Component::IFabric &fabric_, const std::string &fabric_spec_, const std::string ip_address_, std::uint16_t port_);

    remote_memory_client_grouped(remote_memory_client_grouped &&) = default;
    remote_memory_client_grouped &operator=(remote_memory_client_grouped &&) = default;

    ~remote_memory_client_grouped();

    void send_disconnect(Component::IFabric_communicator &cnxn_, registered_memory &rm_, char quit_flag_)
    {
      send_msg(cnxn_, rm_, &quit_flag_, sizeof quit_flag_);
    }

    std::uint64_t vaddr() const { return _vaddr; }
    std::uint64_t key() const { return _key; }
    Component::IFabric_client_grouped &cnxn() { return *_cnxn; }

    Component::IFabric_communicator *allocate_group() const
    {
      return _cnxn->allocate_group();
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
public:
    void check_complete(status_t stat_)
    {
      ASSERT_EQ(stat_, S_OK);
    }
private:
    remote_memory_client_grouped &_parent;
    std::shared_ptr<Component::IFabric_communicator> _cnxn;
    registered_memory _rm_out;
    registered_memory _rm_in;
    registered_memory &rm_out() { return _rm_out; }
    registered_memory &rm_in () { return _rm_in; }
  public:
    remote_memory_subclient(remote_memory_client_grouped &parent_)
      : _parent(parent_)
      , _cnxn(_parent.allocate_group())
      , _rm_out{_parent.cnxn()}
      , _rm_in{_parent.cnxn()}
    {
    }

    Component::IFabric_communicator &cnxn() { return *_cnxn; }

    void write(const std::string &msg_)
    {
      std::copy(msg_.begin(), msg_.end(), &rm_out()[0]);
      std::vector<iovec> buffers(1);
      {
        buffers[0].iov_base = &rm_out()[0];
        buffers[0].iov_len = msg_.size();
        _cnxn->post_write(buffers, _parent.vaddr() + remote_memory_offset, _parent.key(), this);
      }
      wait_poll(
        *_cnxn
        , [this] (void *rmc_, status_t stat_) { check_complete_static_2(this, rmc_, stat_); } /* WAS: check_complete_static */
      );
    }

    void read_verify(const std::string &msg_)
    {
      std::vector<iovec> buffers(1);
      {
        buffers[0].iov_base = &rm_in()[0];
        buffers[0].iov_len = msg_.size();
        _cnxn->post_read(buffers, _parent.vaddr() + remote_memory_offset, _parent.key(), this);
      }
      wait_poll(
        *_cnxn
        , [this] (void *rmc_, status_t stat_) { check_complete_static_2(this, rmc_, stat_); } /* WAS: check_complete_static */
      );
      std::string remote_msg(&rm_in()[0], &rm_in()[0] + msg_.size());
      ASSERT_EQ(msg_, remote_msg);
    }
  };

  void remote_memory_client_grouped::check_complete_static(void *rmc_, status_t stat_)
  {
    auto rmc = static_cast<remote_memory_subclient *>(rmc_);
    ASSERT_TRUE(rmc);
    rmc->check_complete(stat_);
  }

  remote_memory_client_grouped::remote_memory_client_grouped(Component::IFabric &fabric_, const std::string &fabric_spec_, const std::string ip_address_, std::uint16_t port_)
    : _cnxn(open_connection_grouped_patiently(fabric_, fabric_spec_, ip_address_, port_))
    , _rm_out{std::make_shared<registered_memory>(*_cnxn)}
    , _rm_in{std::make_shared<registered_memory>(*_cnxn)}
    , _vaddr{}
    , _key{}
    , _quit_flag('n')
  {
    std::vector<iovec> v;
    iovec iv;
    iv.iov_base = &rm_out()[0];
    iv.iov_len = (sizeof _vaddr) + (sizeof _key);
    v.emplace_back(iv);

   remote_memory_subclient rms(*this);
   auto &cnxn = rms.cnxn();

   cnxn.post_recv(v, this);
   wait_poll(
      cnxn
      , [&v, this] (void *ctxt, status_t st) -> void
        {
          ASSERT_EQ(ctxt, this);
          ASSERT_EQ(st, S_OK);
          ASSERT_EQ(v[0].iov_len, (sizeof _vaddr) + sizeof( _key));
          std::memcpy(&_vaddr, &rm_out()[0], sizeof _vaddr);
          std::memcpy(&_key, &rm_out()[sizeof _vaddr], sizeof _key);
        }
    );
    boost::io::ios_flags_saver sv(std::cerr);
    std::cerr << "Remote memory client addr " << reinterpret_cast<void*>(_vaddr) << " key " << std::hex << _key << std::endl;
  }

  remote_memory_client_grouped::~remote_memory_client_grouped()
  try
  {
    remote_memory_subclient rms(*this);
    auto &cnxn = rms.cnxn();
    send_disconnect(cnxn, rm_out(), _quit_flag);
  }
  catch ( std::exception &e )
  {
    std::cerr << __func__ << " exception " << e.what() << eyecatcher << std::endl;
  }

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

    std::shared_ptr<Component::IFabric_client> _cnxn;
    std::shared_ptr<registered_memory> _rm_out;
    std::shared_ptr<registered_memory> _rm_in;
    std::uint64_t _vaddr;
    std::uint64_t _key;
    char _quit_flag;

    registered_memory &rm_in() const { return *_rm_in; }
    registered_memory &rm_out() const { return *_rm_out; }
  protected:
    void do_quit()
    {
      _quit_flag = 'q';
    }
  public:
    remote_memory_client(Component::IFabric &fabric_, const std::string &fabric_spec_, const std::string ip_address_, std::uint16_t port_)
      : _cnxn(open_connection_patiently(fabric_, fabric_spec_, ip_address_, port_))
      , _rm_out{std::make_shared<registered_memory>(*_cnxn)}
      , _rm_in{std::make_shared<registered_memory>(*_cnxn)}
      , _vaddr{}
      , _key{}
      , _quit_flag('n')
    {
      std::vector<iovec> v;
      iovec iv;
      iv.iov_base = &rm_out()[0];
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
            std::memcpy(&_vaddr, &rm_out()[0], sizeof _vaddr);
            std::memcpy(&_key, &rm_out()[sizeof _vaddr], sizeof _key);
          }
      );
      boost::io::ios_flags_saver sv(std::cerr);
      std::cerr << "Remote memory client addr " << reinterpret_cast<void*>(_vaddr) << " key " << std::hex << _key << std::endl;
    }

    remote_memory_client(remote_memory_client &&) = default;
    remote_memory_client &operator=(remote_memory_client &&) = default;

    void send_disconnect(Component::IFabric_communicator &cnxn_, registered_memory &rm_, char quit_flag_)
    {
      send_msg(cnxn_, rm_, &quit_flag_, sizeof quit_flag_);
    }

    ~remote_memory_client()
    try
    {
      if ( _cnxn )
      {
        send_disconnect(cnxn(), rm_out(), _quit_flag);
      }
    }
    catch ( std::exception &e )
    {
      std::cerr << __func__ << " exception " << e.what() << eyecatcher << std::endl;
    }

    std::uint64_t vaddr() const { return _vaddr; }
    std::uint64_t key() const { return _key; }
    Component::IFabric_client &cnxn() { return *_cnxn; }

    void write(const std::string &msg_)
    {
      std::copy(msg_.begin(), msg_.end(), &rm_out()[0]);
      std::vector<iovec> buffers(1);
      {
        buffers[0].iov_base = &rm_out()[0];
        buffers[0].iov_len = msg_.size();
        _cnxn->post_write(buffers, _vaddr + remote_memory_offset, _key, this);
      }
      wait_poll(
        *_cnxn
        , [this] (void *rmc_, status_t stat_) { check_complete_static_2(this, rmc_, stat_); } /* WAS: check_complete_static */
      );

    }

    void read_verify(const std::string &msg_)
    {
      std::vector<iovec> buffers(1);
      {
        buffers[0].iov_base = &rm_in()[0];
        buffers[0].iov_len = msg_.size();
        _cnxn->post_read(buffers, _vaddr + remote_memory_offset, _key, this);
      }
      wait_poll(
        *_cnxn
        , [this] (void *rmc_, status_t stat_) { check_complete_static_2(this, rmc_, stat_); } /* WAS: check_complete_static */
      );
      std::string remote_msg(&rm_in()[0], &rm_in()[0] + msg_.size());
      ASSERT_EQ(msg_, remote_msg);
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
}

TEST_F(Fabric_test, InstantiateServer)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);

  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));
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
  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));
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
  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));
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
  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));
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
  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));
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
  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));
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
  auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));
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

    auto factory = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));

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

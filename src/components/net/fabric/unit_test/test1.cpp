/* note: we do not include component source, only the API definition */
#include <gtest/gtest.h>
#include <api/components.h>
#include <api/fabric_itf.h>
#if 0
#include <common/utils.h>
#include <core/dpdk.h>
#include <core/physical_memory.h>
#endif

#include <memory>

using namespace Component;

namespace {

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
  static std::shared_ptr<Component::IFabric_factory> _fabric;
};

std::ostream &describe_ep(std::ostream &o_, const IFabric_endpoint &e_)
{
  return o_ << "provider '" << e_.get_provider_name() << "' max_message_size " << e_.max_message_size();
}

#if 0
bool client;
std::shared_ptr<Component::IFabric_factory> Fabric_test::_fabric;
#endif
TEST_F(Fabric_test, InstantiateServer)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);

  ASSERT_TRUE(comp);
  auto fabric = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));

  {
    auto pep1 = std::shared_ptr<Component::IFabric_endpoint>(fabric->open_endpoint("{}"));
    describe_ep(std::cout << "Endpoint 1 ", *pep1) << "\n";
  }
  {
    auto pep2 = std::shared_ptr<Component::IFabric_endpoint>(fabric->open_endpoint("{}"));
    describe_ep(std::cout << "Endpoint 2 ", *pep2) << "\n";
  }
  fabric->release_ref();
}

TEST_F(Fabric_test, InstantiateServerDual)
{
  /* create object instance through factory */
  Component::IBase * comp = Component::load_component("libcomanche-fabric.so",
                                                      Component::net_fabric_factory);

  ASSERT_TRUE(comp);
  auto fabric = std::shared_ptr<Component::IFabric_factory>(static_cast<IFabric_factory *>(comp->query_interface(IFabric_factory::iid())));

  {
    auto pep1 = std::shared_ptr<Component::IFabric_endpoint>(fabric->open_endpoint("{}"));
    auto pep2 = std::shared_ptr<Component::IFabric_endpoint>(fabric->open_endpoint("{}"));
    describe_ep(std::cout << "Endpoint 1 ", *pep1) << "\n";
    describe_ep(std::cout << "Endpoint 2 ", *pep2) << "\n";
  }
  fabric->release_ref();
}

#if 0
std::string remote_host;
#endif

} // namespace

int main(int argc, char **argv) {

#if 0
  if(argc < 2) {
    PINF("fabric-test1 [client ipaddr| server]");
    return -1;
  }

  client = (strcmp(argv[1],"client")==0);
  if(client) {
    assert(argc==3);
    remote_host = argv[2];
  }  
#endif
  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}

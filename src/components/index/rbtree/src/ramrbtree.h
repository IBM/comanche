/*
 * (C) Copyright IBM Corporation 2018. All rights reserved.
 *
 */

/*
 * Authors:
 *
 * Luna Xu
 * Daniel Waddington
 */

#ifndef __RAMRBTREE_COMPONENT_H__
#define __RAMRBTREE_COMPONENT_H__

#include <set>
#include <api/kvindex_itf.h>

using namespace std;

class RamRBTree : public Component::IKVIndex {
 public:
  RamRBTree(const string& owner, const string& name);
  RamRBTree();
  virtual ~RamRBTree();

  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0x8a120985, 0x1253, 0x404d, 0x94d7, 0x77, 0x92, 0x75, 0x21, 0xa1, 0x29); //
  
  void* query_interface(Component::uuid_t& itf_uuid) override
  {
    if (itf_uuid == Component::IKVIndex::iid()) {
      return (void*) static_cast<Component::IKVIndex*>(this);
    }
    else
      return NULL;  // we don't support this interface
  }

  void unload() override { delete this; }

 public:
  virtual void        insert(const std::string& key) override;
  virtual void        erase(const std::string& key) override;
  virtual void        clear() override;
  virtual std::string get(offset_t position) const override;
  virtual size_t      count() const override;
  virtual status_t    find(const std::string& key_expression,
                           offset_t           begin_position,
                           find_t             find_type,
                           offset_t&          out_end_position,
                           std::string&       out_matched_key,
                           unsigned           max_comparisons = 0) override;
private:
  std::set<string> _index;
};

class RamRBTree_factory : public Component::IKVIndex_factory {
 public:
  DECLARE_VERSION(0.1);
  DECLARE_COMPONENT_UUID(0xfac20985, 0x1253, 0x404d, 0x94d7, 0x77, 0x92, 0x75, 0x21, 0xa1, 0x29); //

  void* query_interface(Component::uuid_t& itf_uuid) override
  {
    if (itf_uuid == Component::IKVIndex_factory::iid()) {
      return (void*) static_cast<Component::IKVIndex_factory*>(this);
    }
    else
      return NULL;  // we don't support this interface
  }

  void unload() override { delete this; }

  virtual Component::IKVIndex* create(const std::string& owner,
                                      const std::string& name) override
  {
    Component::IKVIndex* obj =
        static_cast<Component::IKVIndex*>(new RamRBTree(owner, name));
    assert(obj);
    obj->add_ref();
    return obj;
  }
};
#endif

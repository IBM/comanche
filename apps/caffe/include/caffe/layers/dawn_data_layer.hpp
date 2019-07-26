#ifndef CAFFE_DAWN_DATA_LAYER_HPP_
#define CAFFE_DAWN_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "dawn_store.hpp"
#include "kvstoreutility.hpp"
#include "api/kvstore_itf.h"
#include "boost/thread/mutex.hpp"

namespace caffe {

/**
 * @brief Provides data to the Net directly from KVStore.
 *
 */
template <typename Dtype>
class DawnDataLayer : public BaseDataLayer<Dtype> {
 public:
  explicit DawnDataLayer(const LayerParameter& param)
      : BaseDataLayer<Dtype>(param) {}
  virtual ~DawnDataLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*> & bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual inline const char* type() const { return "DawnData"; }
  //virtual inline int ExactNumBottomBlobs() const { return 0; }
  //virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  //virtual void ShuffleImages();
  //virtual void load_batch(Batch<Dtype>* batch);
  virtual void Forward_cpu(const vector<Blob<Dtype>*> & bottom,
	const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*> & bottom,
	const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*> & top,
	const vector<bool> & propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*> & top,
	const vector<bool> & propagate_down, const vector<Blob<Dtype>*>& bottom) {}

  vector<std::pair<std::string, string> > lines_;
  int lines_id_;
  std::string dawn_server_;
  std::string pool_name_;
  std::string pool_dir_;
  std::string device_;
  std::string key_list_file_;
  int batch_size_;
  DawnKVStore_wrapper * kvstore_;
  Pool_wrapper * pool_;
  int height_, width_, channels_;
  unsigned debug_level_;

  // GPU Direct Handles
  Component::IKVStore::memory_handle_t data_handle_;
  Component::IKVStore::memory_handle_t label_handle_;

  static boost::mutex g_lock;


#ifndef CPU_ONLY
//#ifdef USE_KVSTORE

  Blob<Dtype> data_, label_;

//#endif
#endif
};

template <typename Dtype> boost::mutex DawnDataLayer<Dtype>::g_lock;

}  // namespace caffe

#endif  // CAFFE_DAWN_DATA_LAYER_HPP_

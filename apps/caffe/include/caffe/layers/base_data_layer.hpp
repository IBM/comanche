#ifndef CAFFE_DATA_LAYERS_HPP_
#define CAFFE_DATA_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"


#include <iostream>
#include <fstream>

#include "dawn_store.hpp"
#include "gpu_direct_wrapper.h"

namespace caffe {

/**
 * @brief Provides base for data layers that feed blobs to the Net.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class BaseDataLayer : public Layer<Dtype> {
 public:
  explicit BaseDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden except by the BasePrefetchingDataLayer.
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}
  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}

 protected:
  TransformationParameter transform_param_;
  shared_ptr<DataTransformer<Dtype> > data_transformer_;
  bool output_labels_;
};

template <typename Dtype>
class Batch {
 public:
  Blob<Dtype> data_, label_;
};

template <typename Dtype>
class BasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {
 public:
  explicit BasePrefetchingDataLayer(const LayerParameter& param);
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  virtual void InternalThreadEntry();
  virtual void load_batch(Batch<Dtype>* batch) = 0;

  vector<shared_ptr<Batch<Dtype> > > prefetch_;
  BlockingQueue<Batch<Dtype>*> prefetch_free_;
  BlockingQueue<Batch<Dtype>*> prefetch_full_;
  Batch<Dtype>* prefetch_current_;

  Blob<Dtype> transformed_data_;

  /* Used to generate input data for GPU_Direct_Data_Layer*/
  /*
  std::string pool_name_ = "test.pool";
  std::string pool_dir_ = "/mnt/pmem0/";
  std::string key_list_file_ = "key.list";
  KVStore_wrapper * kvstore_;
  int current_key_;
  std::ofstream out_file_;
  const int MAX_DUMP_VALUES = 1000;
  */

  /* Used to generate input data for Dawn_Data_Layer*/
  /*
  std::string pool_name_ = "test.pool";
  std::string pool_dir_ = "/pools";
  std::string key_list_file_ = "key.list";
  std::string dawn_server_ = "10.0.0.51:11900";
  std::string device_ = "mlx5_0";
  unsigned debug_level_ = 0;
  DawnKVStore_wrapper * kvstore_;
  Pool_wrapper * pool_;
  // GPU Direct Handles
  Component::IKVStore::memory_handle_t data_handle_;
  Component::IKVStore::memory_handle_t label_handle_;

  int current_key_;
  std::ofstream out_file_;
  const int MAX_DUMP_VALUES = 1000;
*/
};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_

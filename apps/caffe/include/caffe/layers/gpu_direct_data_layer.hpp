#ifndef CAFFE_GPU_DIRECT_DATA_LAYER_HPP_
#define CAFFE_GPU_DIRECT_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net directly from KVStore.
 *
 */
template <typename Dtype>
class GPUDirectDataLayer : public BaseDataLayer<Dtype> {
 public:
  explicit GPUDirectDataLayer(const LayerParameter& param)
      : BaseDataLayer<Dtype>(param) {}
  virtual ~GPUDirectDataLayer();
  virtual void LayerSetUp(const vector<Blob<Dtype>*> & bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual inline const char* type() const { return "GPUDirectData"; }
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
  std::string pool_name_;
  std::string pool_dir_;
  std::string key_list_file_;
  int batch_size_;
  KVStore_wrapper * kvstore_;
  int height_, width_, channels_;

#ifndef CPU_ONLY
//#ifdef USE_KVSTORE
  GDR_wrapper * gdr_;
  GDR_ptr gdr_ptr_data_;
  gdr_mh_t gdr_mh_data_;
  gdr_info_t gdr_info_data_;

  GDR_ptr gdr_ptr_label_;
  gdr_mh_t gdr_mh_label_;
  gdr_info_t gdr_info_label_;

  Blob<Dtype> data_, label_;

//#endif
#endif
};


}  // namespace caffe

#endif  // CAFFE_GPU_DIRECT_DATA_LAYER_HPP_

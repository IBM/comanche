#include <boost/thread.hpp>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"

#include<iostream>
#include<fstream>

namespace caffe {

template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param)
    : Layer<Dtype>(param),
      transform_param_(param.transform_param()) {
}

template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, this->phase_));
  data_transformer_->InitRand();
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(
    const LayerParameter& param)
    : BaseDataLayer<Dtype>(param),
      prefetch_(param.data_param().prefetch()),
      prefetch_free_(), prefetch_full_(), prefetch_current_() {
  for (int i = 0; i < prefetch_.size(); ++i) {
    prefetch_[i].reset(new Batch<Dtype>());
    prefetch_free_.push(prefetch_[i].get());
  }
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);

  // Before starting the prefetch thread, we make cpu_data and gpu_data
  // calls so that the prefetch thread does not accidentally make simultaneous
  // cudaMalloc calls when the main thread is running. In some GPUs this
  // seems to cause failures if we do not so.
  for (int i = 0; i < prefetch_.size(); ++i) {
    prefetch_[i]->data_.mutable_cpu_data();
    if (this->output_labels_) {
      prefetch_[i]->label_.mutable_cpu_data();
    }
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    for (int i = 0; i < prefetch_.size(); ++i) {
      prefetch_[i]->data_.mutable_gpu_data();
      if (this->output_labels_) {
        prefetch_[i]->label_.mutable_gpu_data();
      }
    }
  }
#endif
  DLOG(INFO) << "Initializing prefetch";
  this->data_transformer_->InitRand();
  StartInternalThread();
  DLOG(INFO) << "Prefetch initialized.";

  /* Used to generate input data for GPU_Direct_Layer*/
  /*
  LOG(INFO) << "SetUp KVStore";
  kvstore_ = new KVStore_wrapper(pool_dir_, pool_name_);
  current_key_ = 0;
  out_file_.open(key_list_file_.c_str(), std::ofstream::out);
  */

  /* Used to generate input data for GPU_Direct_Layer*/
  /*
  LOG(INFO) << "SetUp DawnKVStore";
  pool_ = new Pool_wrapper(pool_dir_, pool_name_);
  kvstore_ = new DawnKVStore_wrapper("public", dawn_server_, device_, debug_level_);
  kvstore_->create_pool(pool_);
  current_key_ = 0;
  out_file_.open(key_list_file_.c_str(), std::ofstream::out);
  */


}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::InternalThreadEntry() {
#ifndef CPU_ONLY
  cudaStream_t stream;
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }
#endif

  try {
    while (!must_stop()) {
      Batch<Dtype>* batch = prefetch_free_.pop();
      load_batch(batch);
#ifndef CPU_ONLY
      if (Caffe::mode() == Caffe::GPU) {
        batch->data_.data().get()->async_gpu_push(stream);
        if (this->output_labels_) {
          batch->label_.data().get()->async_gpu_push(stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
      }
#endif
      prefetch_full_.push(batch);
    }
  } catch (boost::thread_interrupted&) {
    // Interrupted exception is expected on shutdown
  }
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
#endif
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = prefetch_full_.pop("Waiting for data");
  // Reshape to loaded data.
  top[0]->ReshapeLike(prefetch_current_->data_);
  top[0]->set_cpu_data(prefetch_current_->data_.mutable_cpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_current_->label_);
    top[1]->set_cpu_data(prefetch_current_->label_.mutable_cpu_data());
  }
  /* Used for GPU_Direct_Data_Layer */
  /*
  if(current_key_ < MAX_DUMP_VALUES) {
	  std::string data_key = "data_";
	  std::string label_key = "label_";
	  data_key += std::to_string(current_key_);
	  label_key += std::to_string(current_key_);
	  out_file_ << data_key << " " << label_key << std::endl;
	  kvstore_->put(data_key, prefetch_current_->data_.mutable_cpu_data(), prefetch_current_->data_.count()*sizeof(Dtype));
	  kvstore_->put(label_key, prefetch_current_->label_.mutable_cpu_data(), prefetch_current_->label_.count()*sizeof(Dtype));
	  current_key_++;
  }
  */
  /* Used for Dawn_Data_Layer */
  /*
  if(current_key_ < MAX_DUMP_VALUES) {
	  std::string data_key = "data_";
	  std::string label_key = "label_";
	  data_key += std::to_string(current_key_);
	  label_key += std::to_string(current_key_);
	  out_file_ << data_key << " " << label_key << std::endl;
	  data_handle_ = kvstore_ -> register_direct_memory((void*) prefetch_current_->data_.mutable_cpu_data(), prefetch_current_->data_.count()*sizeof(Dtype));
  	  label_handle_ = kvstore_ -> register_direct_memory(prefetch_current_->label_.mutable_cpu_data(), prefetch_current_->label_.count()*sizeof(Dtype));
	  kvstore_->put_direct(pool_, data_key, prefetch_current_->data_.mutable_cpu_data(), prefetch_current_->data_.count()*sizeof(Dtype));
	  kvstore_->put_direct(pool_, label_key, prefetch_current_->label_.mutable_cpu_data(), prefetch_current_->label_.count()*sizeof(Dtype));
	  kvstore_->unregister_direct_memory(data_handle_);
	  kvstore_->unregister_direct_memory(label_handle_);
	  current_key_++;
  }
  */


}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(BasePrefetchingDataLayer, Forward);
#endif

INSTANTIATE_CLASS(BaseDataLayer);
INSTANTIATE_CLASS(BasePrefetchingDataLayer);

}  // namespace caffe

#include <stdint.h>
#include <vector>
#include <string.h>
#include <cuda.h>

#include <sys/mman.h>
#include <unistd.h>
/* Comanche Common */
#include <common/dump_utils.h>
#include <common/utils.h>
#include <api/kvstore_itf.h>

#include "caffe/layers/dawn_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

#include "boost/thread/mutex.hpp"


namespace caffe {

template <typename Dtype>
void DawnDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
#ifndef CPU_ONLY
//#ifdef USE_KVSTORE
  std::string data_key = lines_[lines_id_].first;
  std::string label_key = lines_[lines_id_].second;
  lines_id_++;
  lines_id_ %= lines_.size();

  // Timer
  CPUTimer batch_timer;
  status_t rc;

  LOG(INFO) << "kvstore_get_direct data, key: " << data_key << " label: " << label_key << std::endl;
  batch_timer.Start();
  top[0]->ReshapeLike(data_);
  g_lock.lock();
  rc = kvstore_->get_direct(pool_, data_key, (void *) data_.mutable_gpu_data_gpu_direct(), top[0]->count()*sizeof(Dtype), data_handle_);
  g_lock.unlock();
  if(rc != S_OK)
	  LOG(INFO) << "Error: Data Load";
  top[0]->set_gpu_data((Dtype*)data_.mutable_gpu_data_gpu_direct());
  if(this->output_labels_) {
      top[1]->ReshapeLike(label_);
      g_lock.lock();
      rc = kvstore_->get_direct(pool_, label_key, (void *) label_.mutable_gpu_data_gpu_direct(), top[1]->count()*sizeof(Dtype), label_handle_);
      g_lock.unlock();
      if(rc != S_OK)
    	  LOG(INFO) << "Error: Label Load";
      top[1]->set_gpu_data((Dtype*)label_.mutable_gpu_data_gpu_direct());
  }
  batch_timer.Stop();
  LOG(INFO) << "Load batch: " << batch_timer.MilliSeconds() << " ms.";
  //cudaDeviceSynchronize();
//#endif
#endif

}

INSTANTIATE_LAYER_GPU_FUNCS(DawnDataLayer);

}  // namespace caffe

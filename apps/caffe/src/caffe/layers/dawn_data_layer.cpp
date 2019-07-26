/*
 * dawn_data_layer.cpp
 *
 *  Created on: May 21, 2019
 *      Author: Yanzhao Wu
 */

#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "dawn_store.hpp"
#include "common/utils.h"
#include "caffe/layers/dawn_data_layer.hpp"

#include "core/physical_memory.h"
#include "api/memory_itf.h"
#include <string.h>

/* pmem address translation */
#include "core/xms.h"

#include <boost/algorithm/string.hpp>



namespace caffe {

template <typename Dtype>
DawnDataLayer<Dtype>::~DawnDataLayer<Dtype>() { }

template <typename Dtype>
void DawnDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	// This layer does not transform data.
	CHECK(!this->layer_param_.has_transform_param()) <<
	      this->type() << " does not transform data.";

	if (top.size() == 1) {
		this->output_labels_ = false;
	} else {
		this->output_labels_ = true;
	}
	dawn_server_ = this->layer_param_.dawn_data_param().dawn_server();
	pool_name_ = this->layer_param_.dawn_data_param().pool_name();
	pool_dir_ = this->layer_param_.dawn_data_param().pool_dir();
	device_ = this->layer_param_.dawn_data_param().device();
	key_list_file_ = this->layer_param_.dawn_data_param().key_list_file();
	batch_size_ = this->layer_param_.dawn_data_param().batch_size();
	debug_level_ = this->layer_param_.dawn_data_param().debug_level();

	/* Parse image setting */
	channels_ = this->layer_param_.dawn_data_param().channels();
	height_ = this->layer_param_.dawn_data_param().height();
	width_ = this->layer_param_.dawn_data_param().width();

	LOG(INFO) << "DataLayerSetup: dawn server: " << dawn_server_;
	LOG(INFO) << "DataLayerSetup: device: " << device_;
	LOG(INFO) << "DataLayerSetUp: pool name: " << pool_name_;
	LOG(INFO) << "DataLayerSetUp: pool dir: " << pool_dir_;
	LOG(INFO) << "DataLayerSetUp: key_list_file: " << key_list_file_;
	LOG(INFO) << "DataLayerSetUp: batch size: " << batch_size_;
	pool_ = new Pool_wrapper(pool_dir_, pool_name_);
	kvstore_ = new DawnKVStore_wrapper("public",dawn_server_, device_, debug_level_);
	LOG(INFO) << "Open the pool: " << kvstore_ -> open_pool(pool_);
	LOG(INFO) << "Opening the key list file: " << key_list_file_;
	std::ifstream infile(key_list_file_.c_str());
	std::string line;
	size_t pos;

	std::string data_key;
	std::string label_key;

	while(std::getline(infile, line)) {
		if(line == "" || line.at(0) == '#') {
			continue;
		}
		pos = line.find_first_of(' ');
		if(pos == std::string::npos) {
			continue;
		}
		data_key = line.substr(0,pos);
		boost::trim(data_key);
		label_key = line.substr(pos+1);
		boost::trim(label_key);
		lines_.push_back(std::make_pair(data_key, label_key));
	}
	LOG(INFO) << "A total of " << lines_.size() << " batches.";
	lines_id_ = 0;

	top[0] -> Reshape(batch_size_, channels_, height_, width_);
	LOG(INFO) << "output data size: " << top[0]->num() << ","
	      << top[0]->channels() << "," << top[0]->height() << ","
	      << top[0]->width();
	data_.Reshape(batch_size_, channels_, height_, width_);
	// label
	if(this->output_labels_) {
		vector<int> label_shape(1, batch_size_);
		top[1] -> Reshape(label_shape);
		LOG(INFO) << "label size: " << top[1] -> count() * sizeof(int);
		label_.Reshape(label_shape);
	}

	if(Caffe::mode() == Caffe::CPU) {
		void * data_ptr = (void *) data_.mutable_cpu_data();
		data_handle_ = kvstore_ -> register_direct_memory(data_ptr, data_.count()*sizeof(Dtype));

		if(this->output_labels_) {
			void * label_ptr = (void *) label_.mutable_cpu_data();
			label_handle_ = kvstore_ -> register_direct_memory(label_ptr, label_.count()*sizeof(Dtype));
		}
	} else {
#ifndef CPU_ONLY
	//#ifdef USE_DAWNKVSTORE
		void * data_ptr = (void *) data_.mutable_gpu_data_gpu_direct();
		data_handle_ = kvstore_ -> register_direct_memory(data_ptr, data_.count()*sizeof(Dtype));

		if(this->output_labels_) {
			void * label_ptr = (void *)  label_.mutable_gpu_data_gpu_direct();
			label_handle_ = kvstore_ -> register_direct_memory(label_ptr, label_.count()*sizeof(Dtype));
		}
	//#endif
#else
		LOG(INFO) << "Error: GPU Mode not supported unless CPU_ONLY is enabled";
#endif
	}
}

template <typename Dtype>
void DawnDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  std::string data_key = lines_[lines_id_].first;
  std::string label_key = lines_[lines_id_].second;
  lines_id_++;
  lines_id_ %= lines_.size();

  top[0] -> Reshape(batch_size_, channels_, height_, width_);
  //LOG(INFO) << "kvstore_get_direct data";
  kvstore_->get_direct(pool_, data_key, (void*) top[0]->mutable_cpu_data(), top[0]->count()*sizeof(Dtype), data_handle_);
  if(this->output_labels_) {
      vector<int> label_shape(1, batch_size_);
      top[1] -> Reshape(label_shape);
      //LOG(INFO) << "kvstore_get_direct_label";
      kvstore_->get_direct(pool_, label_key, (void *) top[1]->mutable_cpu_data(), top[1]->count()*sizeof(Dtype), label_handle_);
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(DawnDataLayer, Forward);
#endif

INSTANTIATE_CLASS(DawnDataLayer);
REGISTER_LAYER_CLASS(DawnData);

}  // namespace caffe



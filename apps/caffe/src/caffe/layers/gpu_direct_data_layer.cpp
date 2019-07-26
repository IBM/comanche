/*
 * gpu_direct_data_layer.cpp
 *
 *  Created on: Jun 25, 2018
 *      Author: Yanzhao Wu
 */

#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

#include "gpu_direct_wrapper.h"
#include "common/utils.h"
#include "caffe/layers/gpu_direct_data_layer.hpp"

#include "core/physical_memory.h"
#include "api/memory_itf.h"
#include <string.h>

/* pmem address translation */
#include "core/xms.h"

#include <boost/algorithm/string.hpp>



namespace caffe {

template <typename Dtype>
GPUDirectDataLayer<Dtype>::~GPUDirectDataLayer<Dtype>() { }

// Load data and label from GPUDirect KVStore into the class property blobs.
//template <typename Dtype>
//void GPUDirectDataLayer<Dtype>::LoadHDF5FileData(const char* filename) {
//  DLOG(INFO) << "Loading HDF5 file: " << filename;
//  hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
//  if (file_id < 0) {
//    LOG(FATAL) << "Failed opening HDF5 file: " << filename;
//  }
//
//  int top_size = this->layer_param_.top_size();
//  hdf_blobs_.resize(top_size);
//
//  const int MIN_DATA_DIM = 1;

//
//  for (int i = 0; i < top_size; ++i) {
//    hdf_blobs_[i] = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
//    // Allow reshape here, as we are loading data not params
//    hdf5_load_nd_dataset(file_id, this->layer_param_.top(i).c_str(),
//        MIN_DATA_DIM, MAX_DATA_DIM, hdf_blobs_[i].get(), true);
//  }
//
//  herr_t status = H5Fclose(file_id);
//  CHECK_GE(status, 0) << "Failed to close HDF5 file: " << filename;
//
//  // MinTopBlobs==1 guarantees at least one top blob
//  CHECK_GE(hdf_blobs_[0]->num_axes(), 1) << "Input must have at least 1 axis.";
//  const int num = hdf_blobs_[0]->shape(0);
//  for (int i = 1; i < top_size; ++i) {
//    CHECK_EQ(hdf_blobs_[i]->shape(0), num);
//  }
//  // Default to identity permutation.
//  data_permutation_.clear();
//  data_permutation_.resize(hdf_blobs_[0]->shape(0));
//  for (int i = 0; i < hdf_blobs_[0]->shape(0); i++)
//    data_permutation_[i] = i;
//
//  // Shuffle if needed.
//  if (this->layer_param_.hdf5_data_param().shuffle()) {
//    std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
//    DLOG(INFO) << "Successfully loaded " << hdf_blobs_[0]->shape(0)
//               << " rows (shuffled)";
//  } else {
//    DLOG(INFO) << "Successfully loaded " << hdf_blobs_[0]->shape(0) << " rows";
//  }
//}

template <typename Dtype>
void GPUDirectDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	// This layer does not transform data.
	CHECK(!this->layer_param_.has_transform_param()) <<
	      this->type() << " does not transform data.";

	if (top.size() == 1) {
		this->output_labels_ = false;
	} else {
		this->output_labels_ = true;
	}
	pool_name_ = this->layer_param_.gpu_direct_data_param().pool_name();
	pool_dir_ = this->layer_param_.gpu_direct_data_param().pool_dir();
	key_list_file_ = this->layer_param_.gpu_direct_data_param().key_list_file();
	batch_size_ = this->layer_param_.gpu_direct_data_param().batch_size();

	/* Parse image setting */
	channels_ = this->layer_param_.gpu_direct_data_param().channels();
	height_ = this->layer_param_.gpu_direct_data_param().height();
	width_ = this->layer_param_.gpu_direct_data_param().width();

	LOG(INFO) << "DataLayerSetUp: pool name: " << pool_name_;
	LOG(INFO) << "DataLayerSetUp: pool dir: " << pool_dir_;
	LOG(INFO) << "DataLayerSetUp: key_list_file: " << key_list_file_;
	LOG(INFO) << "DataLayerSetUp: batch size: " << batch_size_;

	kvstore_ = new KVStore_wrapper(pool_dir_, pool_name_);
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

#ifndef CPU_ONLY
//#ifdef USE_KVSTORE
	Core::Physical_memory memory_instance;

	void * bar_ptr_ = NULL;
	void * host_vaddr = NULL;

	addr_t host_paddr = NULL;
	addr_t host_paddr_round_up = NULL;
	void * host_vaddr_xms = NULL;
	unsigned offset_xms = 0;
	gdr_ = new GDR_wrapper();
	gdr_ptr_data_._d_ptr = data_.mutable_gpu_data_gpu_direct();
	//unsigned int flag = 1;
	//CU_CHECK(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, gdr_ptr_data_._d_ptr));
	CHECK(!gdr_->pin_buffer(gdr_ptr_data_._d_ptr,
	    MB(32), 0, 0, &gdr_mh_data_)) << "gdr_pin_buffer data failed";
	CHECK(!gdr_->map(gdr_mh_data_, &bar_ptr_, MB(32))) << "GDR map data failed";
	assert(bar_ptr_);
	CHECK(!gdr_get_info(gdr_->_gdr, gdr_mh_data_, &gdr_info_data_)) << "GDR get_info data failed";
	int offset = ((unsigned long)gdr_ptr_data_._d_ptr) - gdr_info_data_.va;
	LOG(INFO) << "offset of data: " << offset;
	gdr_ptr_data_._h_ptr = (void *) ((char*)bar_ptr_ + offset);
	/*
	host_vaddr =(void *) ((char*)bar_ptr_ + offset);
	host_paddr = xms_get_phys(host_vaddr);
	printf("GDR vaddr=%p paddr=%p\n", (void*)host_vaddr, (void*)host_paddr);
	host_paddr_round_up = round_up(host_paddr, MB(2));
	offset_xms = host_paddr_round_up - host_paddr;
	host_vaddr_xms = ((char *)host_vaddr) + offset_xms;
	host_vaddr_xms = xms_mmap((void*)0x900000000, host_paddr_round_up, MB(2));
	printf("new paddr=0x%lx vaddr=0x%lx\n", (addr_t) host_paddr_round_up, (addr_t)host_vaddr_xms);
        memset(host_vaddr_xms, 0xb, MB(2));
	Component::io_buffer_t iob = memory_instance.register_memory_for_io(host_vaddr_xms, host_paddr_round_up, MB(2));
	gdr_ptr_data_._d_ptr += offset_xms;
	gdr_ptr_data_._h_ptr = host_vaddr_xms;
	*/
	
	if(this->output_labels_) {
	    gdr_ptr_label_._d_ptr = (CUdeviceptr) label_.mutable_gpu_data_gpu_direct();
	    CHECK(!gdr_->pin_buffer(gdr_ptr_label_._d_ptr,
                MB(32), 0, 0, &gdr_mh_label_)) << "gdr_pin_buffer label failed";
            CHECK(!gdr_->map(gdr_mh_label_, &bar_ptr_, MB(32))) << "GDR map label failed";
            assert(bar_ptr_);
            CHECK(!gdr_get_info(gdr_->_gdr, gdr_mh_label_, &gdr_info_label_)) << "GDR get_info label failed";
            int offset = ((unsigned long)gdr_ptr_label_._d_ptr) - gdr_info_label_.va;
            LOG(INFO) << "offset of label: " << offset;
	    gdr_ptr_label_._h_ptr = (void *) ((char*)bar_ptr_ + offset);
	    /*
            host_vaddr =(void *) ((char*)bar_ptr_ + offset);
            host_paddr = xms_get_phys(host_vaddr);
            printf("GDR vaddr=%p paddr=%p\n", (void*)host_vaddr, (void*)host_paddr);
            host_paddr_round_up = round_up(host_paddr, MB(2));
            offset_xms = host_paddr_round_up - host_paddr;
            host_vaddr_xms = ((char*)host_vaddr) + offset_xms;
            host_vaddr_xms = xms_mmap((void*)0x910000000, host_paddr_round_up, MB(2));
            printf("new paddr=0x%lx vaddr=0x%lx\n", (addr_t) host_paddr_round_up, (addr_t)host_vaddr_xms);
	    memset(host_vaddr_xms, 0xb, MB(2));
            //Component::io_buffer_t iob2 = memory_instance.register_memory_for_io(host_vaddr_xms, host_paddr_round_up, MB(2));
            gdr_ptr_label_._d_ptr += offset_xms;
            gdr_ptr_label_._h_ptr = host_vaddr_xms;
	    */
	}

//#endif
#endif



//  // Refuse transformation parameters since HDF5 is totally generic.
//  CHECK(!this->layer_param_.has_transform_param()) <<
//      this->type() << " does not transform data.";
//  // Read the source to parse the filenames.
//  const string& source = this->layer_param_.hdf5_data_param().source();
//  LOG(INFO) << "Loading list of HDF5 filenames from: " << source;
//  hdf_filenames_.clear();
//  std::ifstream source_file(source.c_str());
//  if (source_file.is_open()) {
//    std::string line;
//    while (source_file >> line) {
//      hdf_filenames_.push_back(line);
//    }
//  } else {
//    LOG(FATAL) << "Failed to open source file: " << source;
//  }
//  source_file.close();
//  num_files_ = hdf_filenames_.size();
//  current_file_ = 0;
//  LOG(INFO) << "Number of HDF5 files: " << num_files_;
//  CHECK_GE(num_files_, 1) << "Must have at least 1 HDF5 filename listed in "
//    << source;
//
//  file_permutation_.clear();
//  file_permutation_.resize(num_files_);
//  // Default to identity permutation.
//  for (int i = 0; i < num_files_; i++) {
//    file_permutation_[i] = i;
//  }
//
//  // Shuffle if needed.
//  if (this->layer_param_.hdf5_data_param().shuffle()) {
//    std::random_shuffle(file_permutation_.begin(), file_permutation_.end());
//  }
//
//  // Load the first HDF5 file and initialize the line counter.
//  LoadHDF5FileData(hdf_filenames_[file_permutation_[current_file_]].c_str());
//  current_row_ = 0;
//
//  // Reshape blobs.
//  const int batch_size = this->layer_param_.hdf5_data_param().batch_size();
//  const int top_size = this->layer_param_.top_size();
//  vector<int> top_shape;
//  for (int i = 0; i < top_size; ++i) {
//    top_shape.resize(hdf_blobs_[i]->num_axes());
//    top_shape[0] = batch_size;
//    for (int j = 1; j < top_shape.size(); ++j) {
//      top_shape[j] = hdf_blobs_[i]->shape(j);
//    }
//    top[i]->Reshape(top_shape);
//  }
}

//template <typename Dtype>
//bool HDF5DataLayer<Dtype>::Skip() {
//  int size = Caffe::solver_count();
//  int rank = Caffe::solver_rank();
//  bool keep = (offset_ % size) == rank ||
//              // In test mode, only rank 0 runs, so avoid skipping
//              this->layer_param_.phase() == TEST;
//  return !keep;
//}
//
//template<typename Dtype>
//void HDF5DataLayer<Dtype>::Next() {
//  if (++current_row_ == hdf_blobs_[0]->shape(0)) {
//    if (num_files_ > 1) {
//      ++current_file_;
//      if (current_file_ == num_files_) {
//        current_file_ = 0;
//        if (this->layer_param_.hdf5_data_param().shuffle()) {
//          std::random_shuffle(file_permutation_.begin(),
//                              file_permutation_.end());
//        }
//        DLOG(INFO) << "Looping around to first file.";
//      }
//      LoadHDF5FileData(
//        hdf_filenames_[file_permutation_[current_file_]].c_str());
//    }
//    current_row_ = 0;
//    if (this->layer_param_.hdf5_data_param().shuffle())
//      std::random_shuffle(data_permutation_.begin(), data_permutation_.end());
//  }
//  offset_++;
//}

template <typename Dtype>
void GPUDirectDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  std::string data_key = lines_[lines_id_].first;
  std::string label_key = lines_[lines_id_].second;
  lines_id_++;
  lines_id_ %= lines_.size();

  top[0] -> Reshape(batch_size_, channels_, height_, width_);
  //LOG(INFO) << "kvstore_get_direct data";
  kvstore_->get_direct(data_key, (void*) top[0]->mutable_cpu_data(), top[0]->count()*sizeof(Dtype));
  if(this->output_labels_) {
      vector<int> label_shape(1, batch_size_);
      top[1] -> Reshape(label_shape);
      //LOG(INFO) << "kvstore_get_direct_label";
      kvstore_->get_direct(label_key, (void *) top[1]->mutable_cpu_data(), top[1]->count()*sizeof(Dtype));
  }
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(GPUDirectDataLayer, Forward);
#endif

INSTANTIATE_CLASS(GPUDirectDataLayer);
REGISTER_LAYER_CLASS(GPUDirectData);

}  // namespace caffe



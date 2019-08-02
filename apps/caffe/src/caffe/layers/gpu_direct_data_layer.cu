#include <stdint.h>
#include <vector>
#include <string.h>
#include <cuda.h>

#include <sys/mman.h>
#include <unistd.h>
/* Comanche Common */
#include <common/dump_utils.h>


#include "caffe/layers/gpu_direct_data_layer.hpp"


namespace caffe {

template <typename Dtype>
void GPUDirectDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
#ifndef CPU_ONLY
//#ifdef USE_KVSTORE
  std::string data_key = lines_[lines_id_].first;
  std::string label_key = lines_[lines_id_].second;
  lines_id_++;
  lines_id_ %= lines_.size();

  //LOG(INFO) << "kvstore_get_direct data";
  top[0]->ReshapeLike(data_);

  kvstore_->get_direct(data_key, gdr_ptr_data_._h_ptr, top[0]->count()*sizeof(Dtype));
  //CU_CHECK(cuMemcpyHtoD(gdr_ptr_data_._d_ptr, gdr_ptr_data_._h_ptr, top[0]->count()*sizeof(Dtype)));
  //_mm_sfence();
  //CU_CHECK(cuCtxSynchronize());
  /*  
  void * tmp_d_ptr;
  void * tmp_buffer;
  size_t tmp_buffer_len;
  kvstore_->get(data_key, tmp_buffer, tmp_buffer_len);
  LOG(INFO) << "get length" << tmp_buffer_len;
  //CU_CHECK(cuMemcpyHtoD(gdr_ptr_data_._d_ptr, tmp_buffer, tmp_buffer_len));
  //cudaDeviceSynchronize();

  void * h_tmp = (void *) malloc(top[0]->count()*sizeof(Dtype));
  CUDA_CHECK(cudaMalloc(&tmp_d_ptr, MB(2)));
  copy_kernel<<<1,1024>>>((uint8_t *)gdr_ptr_data_._d_ptr, (uint8_t *) tmp_d_ptr, top[0]->count()*sizeof(Dtype));
  cudaDeviceSynchronize();

  CU_CHECK(cuMemcpyDtoH(h_tmp, (CUdeviceptr) tmp_d_ptr, top[0]->count()*sizeof(Dtype)));
  cudaDeviceSynchronize();

  compare_buf((uint32_t*) h_tmp, (uint32_t*) tmp_buffer, top[0]->count()*sizeof(Dtype));
  printf("Pass GPU copy test\n");
  compare_buf((uint32_t*) gdr_ptr_data_._h_ptr, (uint32_t*) tmp_buffer, top[0]->count()*sizeof(Dtype));
  free(h_tmp);
  free(tmp_buffer);
  */ 
  //_mm_sfence();
  //msync(gdr_ptr_data_._h_ptr, top[0]->count()*sizeof(Dtype) + MB(2), MS_SYNC);
  //sleep(1);
  //LOG(INFO) << "Dump the host pinned GPU memory for Data";
  //hexdump(gdr_ptr_data_._h_ptr, 64);
  //assert(tmp_buffer_len == top[0]->count()*sizeof(Dtype));
  //CU_CHECK(cuMemcpyDtoH(h_tmp, gdr_ptr_data_._d_ptr, top[0]->count()*sizeof(Dtype)));
  //hexdump(h_tmp, 64);
  //compare_buf((uint32_t*) h_tmp, (uint32_t*) tmp_buffer, top[0]->count()*sizeof(Dtype));

  //compare_buf((uint32_t*) h_tmp, (uint32_t*) gdr_ptr_data_._h_ptr, top[0]->count()*sizeof(Dtype));
  //free(h_tmp);

  //kvstore_->get_direct(data_key, (void*) data_.mutable_cpu_data(), top[0]->count()*sizeof(Dtype));
  //hexdump((void *)((char *)tmp_buffer+top[0]->count()*sizeof(Dtype)-256), 256);
  //memcpy(gdr_ptr_data_._h_ptr, tmp_buffer, tmp_buffer_len);
  //CU_CHECK(cuMemcpyHtoD(gdr_ptr_data_._d_ptr, tmp_buffer, tmp_buffer_len));
  //hexdump_kernel<<<1,256>>>((uint8_t*)gdr_ptr_data_._d_ptr, top[0]->count()*sizeof(Dtype)-256);
  //CU_CHECK(cuMemcpyHtoD(gdr_ptr_data_._d_ptr, tmp_buffer, tmp_buffer_len));
  //void* h_tmp1 = (void *) malloc(top[0]->count()*sizeof(Dtype));
  //CU_CHECK(cuMemcpyDtoH(h_tmp1, (CUdeviceptr) data_.mutable_gpu_data(), top[0]->count()*sizeof(Dtype)));
  //compare_buf((uint32_t*) tmp_buffer, (uint32_t*) h_tmp1, top[0]->count()*sizeof(Dtype));
  //compare_buf((uint32_t*) tmp_buffer, (uint32_t*) gdr_ptr_data_._h_ptr, top[0]->count()*sizeof(Dtype));
  top[0]->set_gpu_data((Dtype*)gdr_ptr_data_._d_ptr);
  //free(h_tmp);
  //free(h_tmp1);
  if(this->output_labels_) {
      vector<int> label_shape(1, batch_size_);
      //LOG(INFO) << "kvstore_get_direct_label";
      top[1]->ReshapeLike(label_);
      kvstore_->get_direct(label_key, gdr_ptr_label_._h_ptr, top[1]->count()*sizeof(Dtype));
      //CU_CHECK(cuMemcpyHtoD(gdr_ptr_label_._d_ptr, gdr_ptr_label_._h_ptr, top[1]->count()*sizeof(Dtype)));
      //_mm_sfence();
      //mmiowcwb();
      //CU_CHECK(cuCtxSynchronize());
      //msync(gdr_ptr_label_._h_ptr, top[1]->count()*sizeof(Dtype) + MB(2), MS_SYNC);
      //sleep(1);
      //kvstore_->get(label_key, tmp_buffer, tmp_buffer_len);
      //assert(tmp_buffer_len == top[1]->count()*sizeof(Dtype));

      //h_tmp = (void *) malloc(top[1]->count()*sizeof(Dtype));
      //CU_CHECK(cuMemcpyDtoH(h_tmp, gdr_ptr_label_._d_ptr, top[1]->count()*sizeof(Dtype)));
      //compare_buf((uint32_t*) h_tmp, (uint32_t*) tmp_buffer, top[1]->count()*sizeof(Dtype));
      //compare_buf((uint32_t*) h_tmp, (uint32_t*) gdr_ptr_label_._h_ptr, top[1]->count()*sizeof(Dtype));
      
      //kvstore_->get_direct(label_key, label_.mutable_cpu_data(), top[1]->count()*sizeof(Dtype));
      //CU_CHECK(cuMemcpyHtoD(gdr_ptr_label_._d_ptr, tmp_buffer, tmp_buffer_len));
      //h_tmp1 = (void *) malloc(top[1]->count()*sizeof(Dtype));
      //CU_CHECK(cuMemcpyDtoH(h_tmp1, (CUdeviceptr) label_.mutable_gpu_data(), top[1]->count()*sizeof(Dtype)));
      //compare_buf((uint32_t*) tmp_buffer, (uint32_t*) h_tmp1, top[1]->count()*sizeof(Dtype));
      //LOG(INFO) << "Dump the host pinned GPU memory for Label";
      //hexdump(gdr_ptr_label_._h_ptr, 64);

      //h_tmp = (void *) malloc(top[1]->count()*sizeof(Dtype));
      //CU_CHECK(cuMemcpyDtoH(h_tmp, gdr_ptr_label_._d_ptr, top[1]->count()*sizeof(Dtype)));
      //hexdump(h_tmp, 64);
      //compare_buf((uint32_t*) tmp_buffer, (uint32_t*) gdr_ptr_label_._h_ptr, top[1]->count()*sizeof(Dtype));
      //free(h_tmp);
      //free(h_tmp1);

      top[1]->set_gpu_data((Dtype*)gdr_ptr_label_._d_ptr);
  }
  //cudaDeviceSynchronize();
//#endif
#endif

}

INSTANTIATE_LAYER_GPU_FUNCS(GPUDirectDataLayer);

}  // namespace caffe

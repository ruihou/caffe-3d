#include <vector>

#include "caffe/layers/temporal_max_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void temporal_max_forward(const int nthreads, const Dtype* bottom,
    const int num, const int count, int* max_idx, Dtype* top) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    max_idx[index] = 0;
    top[index] = bottom[index];
    for (int n = 1; n < num; ++n) {
      int bottom_idx = n * count + index;
      if (bottom[bottom_idx] > top[index]) {
        top[index] = bottom[n * count + index];
        max_idx[index] = n;
      }
    }
  }
}

template <typename Dtype>
void TemporalMaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int nthreads = bottom[0]->count(1);
  // NOLINT_NEXT_LINE(whitespace/operators)
  temporal_max_forward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, bottom[0]->gpu_data(), bottom[0]->shape(0), nthreads,
      max_idx_.mutable_gpu_data(), top[0]->mutable_gpu_data());
}

template <typename Dtype>
__global__ void temporal_max_backward(
    const int nthreads, Dtype* bottom, const int num, const int count,
    const int* max_idx, const Dtype* top) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int bottom_index = max_idx[index] * count + index;
    bottom[bottom_index] = top[index];
  }
}

template <typename Dtype>
void TemporalMaxLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int nthreads = bottom[0]->count(1);
  // NOLINT_NEXT_LINE(whitespace/operators)
  temporal_max_backward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, bottom[0]->mutable_gpu_diff(), bottom[0]->shape(0), nthreads,
      max_idx_.gpu_data(), top[0]->gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(TemporalMaxLayer);

}  // namespace caffe

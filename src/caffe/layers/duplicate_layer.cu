#include <vector>

#include "caffe/layers/duplicate_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void DuplicateForward(
    const int nthreads, const Dtype* bottom_data, const int channels_in,
    const int height_in, const int width_in, const int channels_out,
    const int height_out, const int width_out, const int duplicates_,
    Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int num = index;
    const int w = num % width_in;
    num = num / width_in;
    const int h = num % height_in;
    num = num / height_in;
    const int c = num % channels_in;
    num = num / channels_in;

    for (int i = 0; i < duplicates_; ++i) {
      const int out_index = (num * duplicates_ + i) * channels_out
           + c * height_out + h * width_out + w;
      top_data[out_index] = bottom_data[index];
    }
  }
}

template <typename Dtype>
__global__ void DuplicateBackward(
    const int nthreads, Dtype* bottom_data, const int channels_in,
    const int height_in, const int width_in, const int channels_out,
    const int height_out, const int width_out, const int duplicates_,
    const Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int num = index;
    const int w = num % width_in;
    num = num / width_in;
    const int h = num % height_in;
    num = num / height_in;
    const int c = num % channels_in;
    num = num / channels_in;

    bottom_data[index] = 0;
    for (int i = 0; i < duplicates_; ++i) {
      const int out_index = (num * duplicates_ + i) * channels_out
           + c * height_out + h * width_out + w;
      bottom_data[index] += top_data[out_index];
    }
  }
}

template <typename Dtype>
void DuplicateLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = bottom[0]->count();

  // NOLINT_NEXT_LINE(whitespace/operators)
  DuplicateForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom[0]->shape(1), bottom[0]->shape(2),
      bottom[0]->shape(3), top[0]->count(1), top[0]->count(2),
      top[0]->count(3), duplicates_, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void DuplicateLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_data = bottom[0]->mutable_gpu_diff();
  const Dtype* top_data = top[0]->gpu_diff();
  int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  DuplicateBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom[0]->shape(1), bottom[0]->shape(2),
      bottom[0]->shape(3), top[0]->count(1), top[0]->count(2),
      top[0]->count(3), duplicates_, top_data);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(DuplicateLayer);

}  // namespace caffe

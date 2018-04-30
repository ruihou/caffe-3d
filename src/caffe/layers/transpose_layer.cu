#include <vector>

#include "caffe/layers/transpose_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void TransposeForward(
    const int nthreads, const Dtype* bottom_data, const int channels_in,
    const int depth_in, const int height_in, const int width_in,
    const int channels_out, const int height_out,
    const int width_out, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int num = index;
    const int w = num % width_in;
    num = num / width_in;
    const int h = num % height_in;
    num = num / height_in;
    const int d = num % depth_in;
    num = num / depth_in;
    const int c = num % channels_in;
    num = num / channels_in;

    const int out_index = (num * depth_in + d) * channels_out
        + c * height_out
        + h * width_out + w;
    top_data[out_index] = bottom_data[index];
  }
}

template <typename Dtype>
__global__ void TransposeBackward(
    const int nthreads, Dtype* bottom_data, const int channels_in,
    const int depth_in, const int height_in, const int width_in,
    const int channels_out, const int height_out,
    const int width_out, const Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int num = index;
    const int w = num % width_in;
    num = num / width_in;
    const int h = num % height_in;
    num = num / height_in;
    const int d = num % depth_in;
    num = num / depth_in;
    const int c = num % channels_in;
    num = num / channels_in;

    const int out_index = (num * depth_in + d) * channels_out
        + c * height_out
        + h * width_out + w;

    bottom_data[index] = top_data[out_index];
  }
}

template <typename Dtype>
void TransposeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  TransposeForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom[0]->shape(1), bottom[0]->shape(2),
      bottom[0]->shape(3), bottom[0]->shape(4), top[0]->count(1),
      top[0]->count(2), top[0]->count(3), top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void TransposeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_data = bottom[0]->mutable_gpu_diff();
  const Dtype* top_data = top[0]->gpu_diff();
  int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  TransposeBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom[0]->shape(1), bottom[0]->shape(2),
      bottom[0]->shape(3), bottom[0]->shape(4), top[0]->count(1),
      top[0]->count(2), top[0]->count(3), top_data);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(TransposeLayer);

}  // namespace caffe

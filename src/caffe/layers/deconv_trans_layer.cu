#include <vector>

#include "caffe/layers/deconv_trans_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void DeconvTransForward(
    const int nthreads, const Dtype* bottom_data, const int channels_in,
    const int depth_in, const int height_in, const int width_in,
    const int channels_out, const int depth_out, const int height_out,
    const int width_out, const int stride_d, const int stride_h,
    const int stride_w, const int factor, Dtype* top_data) {
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

    int pos = c % factor;
    int i = pos % stride_w;
    pos = pos / stride_w;
    int j = pos % stride_h;
    pos = pos / stride_h;
    int k = pos % stride_d;
    int out_c = c / factor;

    const int out_index = num * channels_out + out_c * depth_out
        + (d * stride_d + k) * height_out
        + (h * stride_h + j) * width_out
        + w * stride_w + i;

    top_data[out_index] = bottom_data[index];
  }
}

template <typename Dtype>
__global__ void DeconvTransBackward(
    const int nthreads, Dtype* bottom_data, const int channels_in,
    const int depth_in, const int height_in, const int width_in,
    const int channels_out, const int depth_out, const int height_out,
    const int width_out, const int stride_d, const int stride_h,
    const int stride_w, const int factor, const Dtype* top_data) {
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

    int pos = c % factor;
    int i = pos % stride_w;
    pos = pos / stride_w;
    int j = pos % stride_h;
    pos = pos / stride_h;
    int k = pos % stride_d;
    int out_c = c / factor;

    const int out_index = num * channels_out + out_c * depth_out
        + (d * stride_d + k) * height_out
        + (h * stride_h + j) * width_out
        + w * stride_w + i;

    bottom_data[index] = top_data[out_index];
  }
}

template <typename Dtype>
void DeconvTransLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  DeconvTransForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom[0]->shape(1), bottom[0]->shape(2),
      bottom[0]->shape(3), bottom[0]->shape(4), top[0]->count(1),
      top[0]->count(2), top[0]->count(3), top[0]->count(4), stride_[0],
      stride_[1], stride_[2], factor_, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void DeconvTransLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_data = bottom[0]->mutable_gpu_diff();
  const Dtype* top_data = top[0]->gpu_diff();
  int count = top[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  DeconvTransBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom[0]->shape(1), bottom[0]->shape(2),
      bottom[0]->shape(3), bottom[0]->shape(4), top[0]->count(1),
      top[0]->count(2), top[0]->count(3), top[0]->count(4), stride_[0],
      stride_[1], stride_[2], factor_, top_data);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(DeconvTransLayer);

}  // namespace caffe

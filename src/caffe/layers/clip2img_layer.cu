#include <vector>

#include "caffe/layers/clip2img_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void clip2img_forward(const int nthreads, const Dtype* bottom,
    const int num, const int channels, const int depth, const int height,
    const int width, Dtype* top) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = index / width % height;
    const int d = index / width / height % depth;
    const int c = index / width / height / depth % channels;
    const int n = index / width / height / depth / channels;

    int out_index = (n * depth + d) * channels * height * width +
        c * height * width + h * width + w;
    top[out_index] = bottom[index];
  }
}

template <typename Dtype>
void Clip2ImgLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int nthreads = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  clip2img_forward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, bottom[0]->gpu_data(), bottom_axes_[0],
      bottom_axes_[1], bottom_axes_[2], bottom_axes_[3], bottom_axes_[4],
      top[0]->mutable_gpu_data());
}

template <typename Dtype>
__global__ void clip2img_backward(
    const int nthreads, Dtype* bottom, const int num, const int channels,
    const int depth, const int height, const int width, const Dtype* top) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % width;
    const int h = index / width % height;
    const int d = index / width / height % depth;
    const int c = index / width / height / depth % channels;
    const int n = index / width / height / depth / channels;

    int top_index = (n * depth + d) * channels * height * width +
        c * height * width + h * width + w;
    bottom[index] = top[top_index];
  }
}

template <typename Dtype>
void Clip2ImgLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int nthreads = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  clip2img_backward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(
      nthreads, bottom[0]->mutable_gpu_diff(), bottom_axes_[0],
      bottom_axes_[1], bottom_axes_[2], bottom_axes_[3], bottom_axes_[4],
      top[0]->gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(Clip2ImgLayer);

}  // namespace caffe

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/normalization_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NormalizationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* squared_data = squared_.mutable_gpu_data();
  Dtype normsqr;
  caffe_gpu_powx(num_item_ * item_count_, bottom_data, Dtype(2), squared_data);
  for (int i = 0; i < num_item_; ++i) {
    caffe_gpu_asum<Dtype>(item_count_, squared_data + i * item_count_, &normsqr);
    caffe_gpu_scale<Dtype>(item_count_, pow(normsqr, -0.5), bottom_data + i * item_count_,
                           top_data + i * item_count_);
  }
}

template <typename Dtype>
void NormalizationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype a;
  for (int i = 0; i < num_item_; ++i) {
    caffe_gpu_dot(item_count_, top_data + i * item_count_,
                  top_diff + i * item_count_, &a);
    caffe_gpu_scale(item_count_, a, top_data + i * item_count_,
                    bottom_diff + i * item_count_);
    caffe_gpu_sub(item_count_, top_diff + i * item_count_,
                  bottom_diff + i * item_count_, bottom_diff + i * item_count_);
    caffe_gpu_dot(item_count_, bottom_data + i * item_count_,
                  bottom_data + i * item_count_, &a);
    caffe_gpu_scale(item_count_, Dtype(pow(a, -0.5)),
                    bottom_diff + i * item_count_,
                    bottom_diff + i * item_count_);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(NormalizationLayer);
}  // namespace caffe

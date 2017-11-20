#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layers/normalization_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NormalizationLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void NormalizationLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape;
  num_item_ = bottom[0]->shape(0);
  item_count_ = bottom[0]->count(1);
  top_shape.push_back(num_item_);
  top_shape.push_back(item_count_);
  top[0]->Reshape(top_shape);
  squared_.ReshapeLike(*top[0]);
}

template <typename Dtype>
void NormalizationLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* squared_data = squared_.mutable_cpu_data();
  caffe_sqr<Dtype>(num_item_ * item_count_, bottom_data, squared_data);
  for (int i = 0; i < num_item_; ++i) {
    Dtype norm_val = caffe_cpu_asum<Dtype>(item_count_,
                                           squared_data + i * item_count_);
    caffe_cpu_scale<Dtype>(item_count_, pow(norm_val, -0.5),
                           bottom_data + i * item_count_,
                           top_data + i * item_count_);
  }
}

template <typename Dtype>
void NormalizationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  for (int i = 0; i < num_item_; ++i) {
    Dtype a = caffe_cpu_dot(item_count_, top_data + i * item_count_,
                            top_diff + i * item_count_);
    caffe_cpu_scale(item_count_, a,
                    top_data + i * item_count_, bottom_diff + i * item_count_);
    caffe_sub(item_count_, top_diff + i * item_count_,
              bottom_diff + i * item_count_, bottom_diff + i * item_count_);
    a = caffe_cpu_dot(item_count_, bottom_data + i * item_count_,
                      bottom_data + i * item_count_);
    caffe_cpu_scale(item_count_, Dtype(pow(a, -0.5)),
                    bottom_diff + i * item_count_,
                    bottom_diff + i * item_count_);
  }
}

#ifdef CPU_ONLY
STUB_GPU(NormalizationLayer);
#endif

INSTANTIATE_CLASS(NormalizationLayer);
REGISTER_LAYER_CLASS(Normalization);

}  // namespace caffe

#include <vector>

#include "caffe/layers/deconv_trans_layer.hpp"

namespace caffe {

template<typename Dtype>
void DeconvTransLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  DeconvTransParameter param = this->layer_param_.deconv_trans_param();
  CHECK_EQ(param.stride_size(), 3) << "Must specify strides in all axis";
  factor_ = 1;
  for (int i = 0; i < 3; ++i) {
    stride_.push_back(param.stride(i));
    factor_ *= param.stride(i);
  }
  CHECK_EQ(bottom[0]->num_axes(), 5) << "Blob must be 5 dim.";
}

template<typename Dtype>
void DeconvTransLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  vector<int> out_shape;
  out_shape.push_back(bottom[0]->shape(0));  // Batch size.
  CHECK_EQ(bottom[0]->shape(1) % factor_, 0)
      << "Number of filters must be the factor of" << factor_ << ".";
  out_shape.push_back(bottom[0]->shape(1) / factor_);  // Channels.
  out_shape.push_back(bottom[0]->shape(2) * stride_[0]);  // Depth.
  out_shape.push_back(bottom[0]->shape(3) * stride_[1]);  // Height.
  out_shape.push_back(bottom[0]->shape(4) * stride_[2]);  // Width.
  top[0]->Reshape(out_shape);
}

template<typename Dtype>
void DeconvTransLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int index = 0; index < bottom[0]->num(); ++index) {
    int num = index;
    const int w = num % bottom[0]->shape(4);
    num = num / bottom[0]->shape(4);
    const int h = num % bottom[0]->shape(3);
    num = num / bottom[0]->shape(3);
    const int d = num % bottom[0]->shape(2);
    num = num / bottom[0]->shape(2);
    const int c = num % bottom[0]->shape(1);
    num = num / bottom[0]->shape(1);

    int pos = c % factor_;
    int i = pos % stride_[2];
    pos = pos / stride_[2];
    int j = pos % stride_[1];
    pos = pos / stride_[1];
    int k = pos % stride_[0];
    int out_c = c / factor_;

    const int out_index = num * top[0]->count(1) + out_c * top[0]->count(2)
        + (d * stride_[0] + k) * top[0]->count(3)
        + (h * stride_[1] + j) * top[0]->count(4)
        + w * stride_[2] + i;
    top_data[out_index] = bottom_data[index];
  }
}

template<typename Dtype>
void DeconvTransLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down,
                                           const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* top_diff = top[0]->cpu_diff();
  for (int index = 0; index < bottom[0]->num(); ++index) {
    int num = index;
    const int w = num % bottom[0]->shape(4);
    num = num / bottom[0]->shape(4);
    const int h = num % bottom[0]->shape(3);
    num = num / bottom[0]->shape(3);
    const int d = num % bottom[0]->shape(2);
    num = num / bottom[0]->shape(2);
    const int c = num % bottom[0]->shape(1);
    num = num / bottom[0]->shape(1);

    int pos = c % factor_;
    int i = pos % stride_[2];
    pos = pos / stride_[2];
    int j = pos % stride_[1];
    pos = pos / stride_[1];
    int k = pos % stride_[0];
    int out_c = c / factor_;

    const int out_index = num * top[0]->count(1) + out_c * top[0]->count(2)
        + (d * stride_[0] + k) * top[0]->count(3)
        + (h * stride_[1] + j) * top[0]->count(4)
        + w * stride_[2] + i;
    bottom_diff[index] = top_diff[out_index];
  }
}

#ifdef CPU_ONLY
STUB_GPU(DeconvTransLayer);
#endif

INSTANTIATE_CLASS(DeconvTransLayer);
REGISTER_LAYER_CLASS(DeconvTrans);

}  // namespace caffe

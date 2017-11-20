#include <vector>

#include "caffe/layers/temporal_max_layer.hpp"

namespace caffe {

template <typename Dtype>
void TemporalMaxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
      "allow in-place computation.";
  CHECK_EQ(bottom[0]->num_axes(), 5) << "Input blob should have dim 5.";
}

template <typename Dtype>
void TemporalMaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape;
  top_shape.push_back(1);
  top_shape.push_back(bottom[0]->count(1));
  top[0]->Reshape(top_shape);
  max_idx_.Reshape(top_shape);
}

template <typename Dtype>
void TemporalMaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void TemporalMaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(TemporalMaxLayer);
#endif

INSTANTIATE_CLASS(TemporalMaxLayer);
REGISTER_LAYER_CLASS(TemporalMax);

}  // namespace caffe

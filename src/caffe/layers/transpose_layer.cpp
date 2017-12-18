#include <vector>

#include "caffe/layers/transpose_layer.hpp"

namespace caffe {

template<typename Dtype>
void TransposeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
}

template<typename Dtype>
void TransposeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                    const vector<Blob<Dtype>*>& top) {
  vector<int> out_shape;
  out_shape.push_back(bottom[0]->shape(0) * bottom[0]->shape(2));  // Batch size.
  out_shape.push_back(bottom[0]->shape(1));  // channels.
  out_shape.push_back(bottom[0]->shape(3));  // height.
  out_shape.push_back(bottom[0]->shape(4));  // width.
  top[0]->Reshape(out_shape);
}

template<typename Dtype>
void TransposeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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

    const int out_index = (num * bottom[0]->shape(2) + d) * top[0]->count(1)
        + c * top[0]->count(2)
        + h * top[0]->count(3) + w;
    top_data[out_index] = bottom_data[index];
  }
}

template<typename Dtype>
void TransposeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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

    const int out_index = (num * bottom[0]->shape(2) + d) * top[0]->count(1)
        + c * top[0]->count(2)
        + h * top[0]->count(3) + w;
    bottom_diff[index] = top_diff[out_index];
  }
}

#ifdef CPU_ONLY
STUB_GPU(TransposeLayer);
#endif

INSTANTIATE_CLASS(TransposeLayer);
REGISTER_LAYER_CLASS(Transpose);

}  // namespace caffe

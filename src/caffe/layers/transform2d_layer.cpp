#include <vector>

#include "caffe/layers/transform2d_layer.hpp"

namespace caffe {

template<typename Dtype>
void Transform2DLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num_axes(), 4) << "Blob must be 4 dim.";
}

template<typename Dtype>
void Transform2DLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                      const vector<Blob<Dtype>*>& top) {
  vector<int> out_shape;
  out_shape.push_back(bottom[0]->shape(0));  // Batch size.
  out_shape.push_back(bottom[0]->shape(1) * 4);  // channels.
  out_shape.push_back(bottom[0]->shape(2) / 2);  // height.
  out_shape.push_back(bottom[0]->shape(3) / 2);  // height.
  top[0]->Reshape(out_shape);
}

template<typename Dtype>
void Transform2DLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                          const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  for (int index = 0; index < bottom[0]->count(); ++index) {
    int num = index;
    const int w = num % bottom[0]->shape(3);
    num = num / bottom[0]->shape(3);
    const int h = num % bottom[0]->shape(2);
    num = num / bottom[0]->shape(2);
    const int c = num % bottom[0]->shape(1);
    num = num / bottom[0]->shape(1);

    int i = h % 2;
    int out_h = h / 2;
    int j = w % 2;
    int out_w = w / 2;
    int out_c = c + (i * 2 + j) * bottom[0]->shape(1);

    const int out_index = num * top[0]->count(1) + out_c * top[0]->count(2)
        + out_h * top[0]->count(3) + out_w;
    top_data[out_index] = bottom_data[index];
  }
}

template<typename Dtype>
void Transform2DLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                           const vector<bool>& propagate_down,
                                           const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* top_diff = top[0]->cpu_diff();
  for (int index = 0; index < bottom[0]->count(); ++index) {
    int num = index;
    const int w = num % bottom[0]->shape(3);
    num = num / bottom[0]->shape(3);
    const int h = num % bottom[0]->shape(2);
    num = num / bottom[0]->shape(2);
    const int c = num % bottom[0]->shape(1);
    num = num / bottom[0]->shape(1);

    int i = h % 2;
    int out_h = h / 2;
    int j = w % 2;

    int out_w = w / 2;
    int out_c = c + (i * 2 + j) * bottom[0]->shape(1);

    const int out_index = num * top[0]->count(1) + out_c * top[0]->count(2)
        + out_h * top[0]->count(3) + out_w;
    bottom_diff[index] = top_diff[out_index];
  }
}

#ifdef CPU_ONLY
STUB_GPU(Transform2DLayer);
#endif

INSTANTIATE_CLASS(Transform2DLayer);
REGISTER_LAYER_CLASS(Transform2D);

}  // namespace caffe

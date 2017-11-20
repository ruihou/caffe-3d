#include <vector>

#include "caffe/layers/clip2img_layer.hpp"

namespace caffe {

template <typename Dtype>
void Clip2ImgLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                     const vector<Blob<Dtype>*>& top) {
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
      "allow in-place computation.";
  CHECK_EQ(bottom[0]->num_axes(), 5) << "Input blob should have dim 5.";
}

template <typename Dtype>
void Clip2ImgLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                   const vector<Blob<Dtype>*>& top) {
  bottom_axes_ = bottom[0]->shape();
  top[0]->Reshape(bottom_axes_[0] * bottom_axes_[2],
                  bottom_axes_[1], bottom_axes_[3], bottom_axes_[4]);
}

template <typename Dtype>
void Clip2ImgLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void Clip2ImgLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(Clip2ImgLayer);
#endif

INSTANTIATE_CLASS(Clip2ImgLayer);
REGISTER_LAYER_CLASS(Clip2Img);

}  // namespace caffe

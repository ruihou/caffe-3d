#ifdef USE_CUDNN
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/cudnn_softmax_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNSoftmaxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  SoftmaxLayer<Dtype>::LayerSetUp(bottom, top);
  // Initialize CUDNN.
  CUDNN_CHECK(cudnnCreate(&handle_));
  cudnn::createTensorDesc<Dtype>(&bottom_desc_);
  cudnn::createTensorDesc<Dtype>(&top_desc_);
  handles_setup_ = true;
}

template <typename Dtype>
void CuDNNSoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  SoftmaxLayer<Dtype>::Reshape(bottom, top);
  cudnn::setTensorNdDesc<Dtype>(&bottom_desc_, bottom[0]->shape());
  cudnn::setTensorNdDesc<Dtype>(&top_desc_, bottom[0]->shape());
}

template <typename Dtype>
CuDNNSoftmaxLayer<Dtype>::~CuDNNSoftmaxLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);
  cudnnDestroy(handle_);
}

INSTANTIATE_CLASS(CuDNNSoftmaxLayer);

}  // namespace caffe
#endif

#include <cmath>
#include <vector>

#include "caffe/layers/aug_sigmoid_layer.hpp"

namespace caffe {

template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 0.5 * tanh(0.5 * x) + 0.5;
}

template <typename Dtype>
void AugSigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype multiplier = this->layer_param_.aug_sigmoid_param().multiplier();
  for (int i = 0; i < count; ++i) {
    top_data[i] = sigmoid(bottom_data[i]) * multiplier;
  }
}

template <typename Dtype>
void AugSigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype multiplier = this->layer_param_.aug_sigmoid_param().multiplier();
    for (int i = 0; i < count; ++i) {
      const Dtype sigmoid_x = top_data[i] / multiplier;
      bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x) * multiplier;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(AugSigmoidLayer);
#endif

INSTANTIATE_CLASS(AugSigmoidLayer);
REGISTER_LAYER_CLASS(AugSigmoid);

}  // namespace caffe

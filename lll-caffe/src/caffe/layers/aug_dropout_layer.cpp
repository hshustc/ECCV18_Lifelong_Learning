// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/layers/aug_dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void AugDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
  scale_ = 1. / (1. - threshold_);
  uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
  assigned_mask_ = this->layer_param_.aug_dropout_param().assigned_mask();
  if(assigned_mask_){
      CHECK(bottom.size() == 2 && top.size() == 1);
  }
}

template <typename Dtype>
void AugDropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  // Set up the cache for random number generation
  // ReshapeLike does not work because rand_vec_ is of Dtype uint
  rand_vec_.Reshape(bottom[0]->shape());
  if(top.size() > 1){
      top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void AugDropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  unsigned int* mask = rand_vec_.mutable_cpu_data();
  const Dtype* assigned_mask=NULL;
  Dtype* top_mask_data=NULL;
  if(assigned_mask_){
      assigned_mask = bottom[1]->cpu_data();
  }
  if(top.size() > 1){
      top_mask_data = top[1]->mutable_cpu_data();
  }

  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    // Create random numbers
    caffe_rng_bernoulli(count, 1. - threshold_, mask);
    for (int i = 0; i < count; ++i) {
      //top_data[i] = bottom_data[i] * mask[i] * scale_;
      if(assigned_mask_){
          top_data[i] = bottom_data[i] * static_cast<unsigned int>(assigned_mask[i]) * scale_;
      }
      else{
          top_data[i] = bottom_data[i] * mask[i] * scale_;
          if(top.size() > 1){
              top_mask_data[i] = static_cast<Dtype>(mask[i]);
          }
      }
    }
  } else {
    caffe_copy(bottom[0]->count(), bottom_data, top_data);
  }
}

template <typename Dtype>
void AugDropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  //if(bottom.size() == 2 && propagate_down[1]){
      //LOG(FATAL)<<this->type()
          //<<"Layer cannot backpropagate to assigned_mask";
  //}
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    if (this->phase_ == TRAIN) {
      const unsigned int* mask = rand_vec_.cpu_data();
      const int count = bottom[0]->count();
      const Dtype* assigned_mask=NULL;
      if(assigned_mask_){
          assigned_mask = bottom[1]->cpu_data();
      }
      for (int i = 0; i < count; ++i) {
        if(assigned_mask_){
            bottom_diff[i] = top_diff[i] * static_cast<unsigned int>(assigned_mask[i]) * scale_;
        }
        else{
            bottom_diff[i] = top_diff[i] * mask[i] * scale_;
        }
      }
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(AugDropoutLayer);
#endif

INSTANTIATE_CLASS(AugDropoutLayer);
REGISTER_LAYER_CLASS(AugDropout);

}  // namespace caffe

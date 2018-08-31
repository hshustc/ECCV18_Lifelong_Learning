#include <vector>

#include "caffe/layers/aug_dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void AugDropoutForward(const int n, const Dtype* in,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] * (mask[index] > threshold) * scale;
  }
}

template <typename Dtype>
__global__ void AugDropoutForward_Copymask(const int n, const Dtype* in,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = static_cast<Dtype>(mask[index]);
  }
}

template <typename Dtype>
__global__ void AugDropoutForwardV2(const int n, const Dtype* in,
    const Dtype* assigned_mask, const unsigned int threshold, const float scale,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index] * (static_cast<unsigned int>(assigned_mask[index]) > threshold) * scale;
  }
}

template <typename Dtype>
void AugDropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  if (this->phase_ == TRAIN) {
    unsigned int* mask =
        static_cast<unsigned int*>(rand_vec_.mutable_gpu_data());
    caffe_gpu_rng_uniform(count, mask);

    const Dtype* assigned_mask=NULL;
    Dtype* top_mask_data=NULL;
    if(assigned_mask_){
        assigned_mask = bottom[1]->gpu_data();
    }
    if(top.size() > 1){
        top_mask_data = top[1]->mutable_gpu_data();
    }

    // set thresholds
    // NOLINT_NEXT_LINE(whitespace/operators)
    if(assigned_mask_){
        AugDropoutForwardV2<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_data, assigned_mask, uint_thres_, scale_, top_data);
    }
    else{
        AugDropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, bottom_data, mask, uint_thres_, scale_, top_data);
        if(top.size() > 1){
            AugDropoutForward_Copymask<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
                count, bottom_data, mask, uint_thres_, scale_, top_mask_data);
        }
    }
    CUDA_POST_KERNEL_CHECK;
  } else {
    caffe_copy(count, bottom_data, top_data);
  }
}

template <typename Dtype>
__global__ void AugDropoutBackward(const int n, const Dtype* in_diff,
    const unsigned int* mask, const unsigned int threshold, const float scale,
    Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * scale * (mask[index] > threshold);
  }
}

template <typename Dtype>
__global__ void AugDropoutBackwardV2(const int n, const Dtype* in_diff,
    const Dtype* assigned_mask, const unsigned int threshold, const float scale,
    Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * scale * (static_cast<unsigned int>(assigned_mask[index]) > threshold);
  }
}

template <typename Dtype>
void AugDropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    if (this->phase_ == TRAIN) {
      const unsigned int* mask =
          static_cast<const unsigned int*>(rand_vec_.gpu_data());
      const int count = bottom[0]->count();
      const Dtype* assigned_mask=NULL;
      if(assigned_mask_){
          assigned_mask = bottom[1]->gpu_data();
      }
      // NOLINT_NEXT_LINE(whitespace/operators)
      if(assigned_mask_){
          AugDropoutBackwardV2<Dtype><<<CAFFE_GET_BLOCKS(count),
            CAFFE_CUDA_NUM_THREADS>>>(
              count, top_diff, assigned_mask, uint_thres_, scale_, bottom_diff);
      }
      else{
          AugDropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
            CAFFE_CUDA_NUM_THREADS>>>(
              count, top_diff, mask, uint_thres_, scale_, bottom_diff);
      }
      CUDA_POST_KERNEL_CHECK;
    } else {
      caffe_copy(top[0]->count(), top_diff, bottom_diff);
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(AugDropoutLayer);

}  // namespace caffe

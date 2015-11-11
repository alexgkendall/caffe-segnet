#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class SoftmaxWithLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  SoftmaxWithLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 2, 3)),
        blob_bottom_label_(new Blob<Dtype>(10, 1, 2, 3)),
        blob_top_loss_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    for (int i = 0; i < blob_bottom_label_->count(); ++i) {
      blob_bottom_label_->mutable_cpu_data()[i] = caffe_rng_rand() % 5;
    }
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~SoftmaxWithLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_loss_;
  }
  void SetUpDense(Dtype black_presoftmax_pos, Dtype black_presoftmax_neg,
                  Dtype white_presoftmax_pos, Dtype white_presoftmax_neg) {
    /*
      label: 10 examples of 1 channel:
      [1 1 1 1]
      [0 0 0 0]
    */
    vector<int> label_shape;
    label_shape.push_back(10);
    label_shape.push_back(1);
    label_shape.push_back(2);
    label_shape.push_back(4);
    blob_bottom_label_->Reshape(label_shape);
    Dtype* label = blob_bottom_label_->mutable_cpu_data();
    for (int n = 0; n < label_shape[0]; ++n) {
      for (int c = 0; c < label_shape[1]; ++c) {
        for (int h = 0; h < label_shape[2]; ++h) {
          for (int w = 0; w < label_shape[3]; ++w) {
            label[blob_bottom_label_->offset(n, c, h, w)] = 1 - h;
          }
        }
      }
    }
    /*
      data: 10 examples of 2 channel:
      [[w- w- w- w-] , [w+ w+ w+ w+]
       [b+ b+ b+ b+]   [b- b- b- b-]]
    */
    vector<int> shape;
    shape.push_back(10);
    shape.push_back(2);
    shape.push_back(2);
    shape.push_back(4);
    blob_bottom_data_->Reshape(shape);
    Dtype* data = blob_bottom_data_->mutable_cpu_data();
    for (int n = 0; n < shape[0]; ++n) {
      for (int c = 0; c < shape[1]; ++c) {
        for (int h = 0; h < shape[2]; ++h) {
          Dtype val;
          if (c) {
            val = h ? black_presoftmax_neg : white_presoftmax_pos;
          } else {
            val = h ? black_presoftmax_pos : white_presoftmax_neg;
          }
          for (int w = 0; w < shape[3]; ++w) {
            data[blob_bottom_data_->offset(n,c,h,w)] = val;
          }
        }
      }
    }
  }
  void SetUpImbalanced() {
    this->blob_bottom_data_->Reshape(10, 8, 3, 4);
    this->blob_bottom_label_->Reshape(10, 1, 3, 4);
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    Dtype* labels = blob_bottom_label_->mutable_cpu_data();
    const int label_count = blob_bottom_label_->count();
    for (int i = 0; i < label_count; ++i) {
      labels[i] = std::rand() % 8;
    }
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SoftmaxWithLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(SoftmaxWithLossLayerTest, TestLoss) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_loss_weight(1);
  SoftmaxWithLossLayer<Dtype> layer(layer_param);
  this->SetUpDense(1,-1,1,-1);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype full_loss = this->blob_top_loss_->cpu_data()[0];
  this->SetUpDense(2,-1,1,-1);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype better_loss = this->blob_top_loss_->cpu_data()[0];
  EXPECT_LT(better_loss, full_loss);
}

TYPED_TEST(SoftmaxWithLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  SoftmaxWithLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxWithLossLayerTest, TestFrequencyWeightedGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  LossParameter* loss_param = layer_param.mutable_loss_param();
  for(int i=0; i<8; i++) {
    loss_param->add_class_weighting(2);
  }
  loss_param->set_weight_by_label_freqs(true);
  SoftmaxWithLossLayer<Dtype> layer(layer_param);
  this->SetUpImbalanced();
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxWithLossLayerTest, TestForwardIgnoreLabel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_loss_param()->set_normalize(false);
  // First, compute the loss with all labels
  scoped_ptr<SoftmaxWithLossLayer<Dtype> > layer(
      new SoftmaxWithLossLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  Dtype full_loss = this->blob_top_loss_->cpu_data()[0];
  // Now, accumulate the loss, ignoring each label in {0, ..., 4} in turn.
  Dtype accum_loss = 0;
  for (int label = 0; label < 5; ++label) {
    layer_param.mutable_loss_param()->set_ignore_label(label);
    layer.reset(new SoftmaxWithLossLayer<Dtype>(layer_param));
    layer->SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer->Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    accum_loss += this->blob_top_loss_->cpu_data()[0];
  }
  // Check that each label was included all but once.
  EXPECT_NEAR(4 * full_loss, accum_loss, 1e-4);
}

TYPED_TEST(SoftmaxWithLossLayerTest, TestGradientIgnoreLabel) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  // labels are in {0, ..., 4}, so we'll ignore about a fifth of them
  layer_param.mutable_loss_param()->set_ignore_label(0);
  SoftmaxWithLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

TYPED_TEST(SoftmaxWithLossLayerTest, TestGradientUnnormalized) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_loss_param()->set_normalize(false);
  SoftmaxWithLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe

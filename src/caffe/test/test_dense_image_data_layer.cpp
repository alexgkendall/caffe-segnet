#include <map>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class DenseImageDataLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  DenseImageDataLayerTest()
      : seed_(1701),
        blob_top_data_(new Blob<Dtype>()),
        blob_top_label_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    blob_top_vec_.push_back(blob_top_data_);
    blob_top_vec_.push_back(blob_top_label_);
    Caffe::set_random_seed(seed_);
    // Create test input file.
    MakeTempFilename(&filename_);
    std::ofstream outfile(filename_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_;
    for (int i = 0; i < 5; ++i) {
      outfile << EXAMPLES_SOURCE_DIR "images/cat.jpg " 
              << EXAMPLES_SOURCE_DIR "images/cat_label.png ";
    }
    outfile.close();
    // Create test input file for images of distinct sizes.
    MakeTempFilename(&filename_reshape_);
    std::ofstream reshapefile(filename_reshape_.c_str(), std::ofstream::out);
    LOG(INFO) << "Using temporary file " << filename_reshape_;
    reshapefile << EXAMPLES_SOURCE_DIR "images/cat.jpg " 
        << EXAMPLES_SOURCE_DIR "images/cat_label.png ";
    reshapefile << EXAMPLES_SOURCE_DIR "images/fish-bike.jpg " 
        << EXAMPLES_SOURCE_DIR "images/fish-bike_label.png ";
    reshapefile.close();
  }

  virtual ~DenseImageDataLayerTest() {
    delete blob_top_data_;
    delete blob_top_label_;
  }

  int seed_;
  string filename_;
  string filename_reshape_;
  Blob<Dtype>* const blob_top_data_;
  Blob<Dtype>* const blob_top_label_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(DenseImageDataLayerTest, TestDtypesAndDevices);

TYPED_TEST(DenseImageDataLayerTest, TestRead) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  DenseImageDataParameter* image_data_param = param.mutable_dense_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_.c_str());
  image_data_param->set_shuffle(false);
  DenseImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 360);
  EXPECT_EQ(this->blob_top_label_->width(), 480);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    int min_label_val = 100;
    int max_label_val = -1;
    Dtype sum = 0;
    for (int i = 0; i < this->blob_top_label_->count(); ++i) {
      int label_val = static_cast<int>(this->blob_top_label_->cpu_data()[i]);
      sum += label_val;
      min_label_val = std::min(min_label_val, label_val);
      max_label_val = std::max(max_label_val, label_val);
      EXPECT_TRUE(0 <= label_val && label_val <= 2);
    }
    EXPECT_NEAR(sum / this->blob_top_label_->count(), 1.475104, 1e-4);
    EXPECT_EQ(min_label_val, 0);
    EXPECT_EQ(max_label_val, 2);
  }
}

TYPED_TEST(DenseImageDataLayerTest, TestResize) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  DenseImageDataParameter* image_data_param = param.mutable_dense_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_.c_str());
  image_data_param->set_new_height(256);
  image_data_param->set_new_width(256);
  image_data_param->set_shuffle(false);
  DenseImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 256);
  EXPECT_EQ(this->blob_top_data_->width(), 256);
  EXPECT_EQ(this->blob_top_label_->num(), 5);
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), 256);
  EXPECT_EQ(this->blob_top_label_->width(), 256);
  // Go through the data twice
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    int min_label_val = 100;
    int max_label_val = -1;
    for (int i = 0; i < this->blob_top_label_->count(); ++i) {
      int label_val = static_cast<int>(this->blob_top_label_->cpu_data()[i]);
      min_label_val = std::min(min_label_val, label_val);
      max_label_val = std::max(max_label_val, label_val);
      EXPECT_TRUE(0 <= label_val && label_val <= 2);
    }
    EXPECT_EQ(min_label_val, 0);
    EXPECT_EQ(max_label_val, 2);
  }
}

TYPED_TEST(DenseImageDataLayerTest, TestReshape) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  DenseImageDataParameter* image_data_param = param.mutable_dense_image_data_param();
  image_data_param->set_batch_size(1);
  image_data_param->set_source(this->filename_reshape_.c_str());
  image_data_param->set_shuffle(false);
  DenseImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_label_->num(), this->blob_top_data_->num());
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), this->blob_top_data_->height());
  EXPECT_EQ(this->blob_top_label_->width(), this->blob_top_data_->width());
  // cat.jpg
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 1);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->num(), this->blob_top_data_->num());
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), this->blob_top_data_->height());
  EXPECT_EQ(this->blob_top_label_->width(), this->blob_top_data_->width());
  // fish-bike.jpg
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 1);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 323);
  EXPECT_EQ(this->blob_top_data_->width(), 481);
  EXPECT_EQ(this->blob_top_label_->num(), this->blob_top_data_->num());
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), this->blob_top_data_->height());
  EXPECT_EQ(this->blob_top_label_->width(), this->blob_top_data_->width());
}

TYPED_TEST(DenseImageDataLayerTest, TestShuffle) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter param;
  DenseImageDataParameter* image_data_param = param.mutable_dense_image_data_param();
  image_data_param->set_batch_size(5);
  image_data_param->set_source(this->filename_.c_str());
  image_data_param->set_shuffle(true);
  DenseImageDataLayer<Dtype> layer(param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_data_->num(), 5);
  EXPECT_EQ(this->blob_top_data_->channels(), 3);
  EXPECT_EQ(this->blob_top_data_->height(), 360);
  EXPECT_EQ(this->blob_top_data_->width(), 480);
  EXPECT_EQ(this->blob_top_label_->num(), this->blob_top_data_->num());
  EXPECT_EQ(this->blob_top_label_->channels(), 1);
  EXPECT_EQ(this->blob_top_label_->height(), this->blob_top_data_->height());
  EXPECT_EQ(this->blob_top_label_->width(), this->blob_top_data_->width());
  // Go through the data twice
  /*
  for (int iter = 0; iter < 2; ++iter) {
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    map<Dtype, int> values_to_indices;
    int num_in_order = 0;
    for (int i = 0; i < 5; ++i) {
      Dtype value = this->blob_top_label_->cpu_data()[i];
      // Check that the value has not been seen already (no duplicates).
      EXPECT_EQ(values_to_indices.find(value), values_to_indices.end());
      values_to_indices[value] = i;
      num_in_order += (value == Dtype(i));
    }
    EXPECT_EQ(5, values_to_indices.size());
    EXPECT_GT(5, num_in_order);
  }
  */
}

}  // namespace caffe

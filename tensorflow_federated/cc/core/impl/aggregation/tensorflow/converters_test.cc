/*
 * Copyright 2022 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorflow_federated/cc/core/impl/aggregation/tensorflow/converters.h"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// clang-format off
#include "tensorflow_federated/cc/core/impl/aggregation/testing/parse_text_proto.h"
// clang-format on
#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/protobuf/struct.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_spec.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/vector_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/testing.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"

namespace tensorflow_federated::aggregation::tensorflow {
namespace {

namespace tf = ::tensorflow;

tf::TensorShape CreateTfShape(std::initializer_list<int64_t> dim_sizes) {
  tf::TensorShape shape;
  EXPECT_OK(tf::TensorShape::BuildTensorShape(dim_sizes, &shape));
  return shape;
}

tf::PartialTensorShape CreatePartialTfShape(
    std::initializer_list<int64_t> dim_sizes) {
  tf::PartialTensorShape shape;
  tf::TensorShapeProto shape_proto;
  for (auto dim_size : dim_sizes) {
    shape_proto.add_dim()->set_size(dim_size);
  }
  EXPECT_OK(
      tf::PartialTensorShape::BuildPartialTensorShape(shape_proto, &shape));
  return shape;
}

tf::TensorSpecProto CreateTfTensorSpec(
    const std::string& name, tf::DataType dtype,
    std::initializer_list<int64_t> dim_sizes) {
  tf::TensorSpecProto spec;
  spec.set_name(name);
  spec.set_dtype(dtype);
  for (auto dim_size : dim_sizes) {
    spec.mutable_shape()->add_dim()->set_size(dim_size);
  }
  return spec;
}

TEST(ConvertersTest, ConvertsSupportedTfDataTypeToAggDataType) {
  EXPECT_EQ(*ToAggDataType(tf::DT_FLOAT), DT_FLOAT);
  EXPECT_EQ(*ToAggDataType(tf::DT_DOUBLE), DT_DOUBLE);
  EXPECT_EQ(*ToAggDataType(tf::DT_INT32), DT_INT32);
  EXPECT_EQ(*ToAggDataType(tf::DT_INT64), DT_INT64);
  EXPECT_EQ(*ToAggDataType(tf::DT_STRING), DT_STRING);
}

TEST(ConvertersTest, CannotConvertUnsupportedTfDataTypeToAggDataType) {
  EXPECT_THAT(ToAggDataType(tf::DT_VARIANT), StatusIs(INVALID_ARGUMENT));
}

TEST(ConvertersTest, ConvertsSupportedTfShapeToAggShape) {
  EXPECT_EQ(ToAggShape(CreateTfShape({})), TensorShape({}));
  EXPECT_EQ(ToAggShape(CreateTfShape({1})), TensorShape({1}));
  EXPECT_EQ(ToAggShape(CreateTfShape({2, 3})), TensorShape({2, 3}));
}

TEST(ConvertersTest, ConvertsSupportedTfPartialShapeToAggShape) {
  EXPECT_EQ(ToAggShape(CreatePartialTfShape({})), TensorShape({}));
  EXPECT_EQ(ToAggShape(CreatePartialTfShape({-1})), TensorShape({-1}));
  EXPECT_EQ(ToAggShape(CreatePartialTfShape({2, -1})), TensorShape({2, -1}));
  // All negative dimensions are interpreted by tensorflow the same way as -1.
  EXPECT_EQ(ToAggShape(CreatePartialTfShape({2, -3})), TensorShape({2, -1}));
}

TEST(ConvertersTest, ConvertsTfTensorSpecToAggTensorSpec) {
  auto tensor_spec =
      ToAggTensorSpec(CreateTfTensorSpec("foo", tf::DT_FLOAT, {1, 2, 3}));
  ASSERT_THAT(tensor_spec, IsOk());
  EXPECT_EQ(tensor_spec->name(), "foo");
  EXPECT_EQ(tensor_spec->dtype(), DT_FLOAT);
  EXPECT_EQ(tensor_spec->shape(), TensorShape({1, 2, 3}));
}

TEST(ConvertersTest, ConvertsTfTensorSpecWithUnknownDimensionToAggTensorSpec) {
  auto tensor_spec =
      ToAggTensorSpec(CreateTfTensorSpec("foo", tf::DT_FLOAT, {1, -1}));
  ASSERT_THAT(tensor_spec, IsOk());
  EXPECT_EQ(tensor_spec->name(), "foo");
  EXPECT_EQ(tensor_spec->dtype(), DT_FLOAT);
  EXPECT_EQ(tensor_spec->shape(), TensorShape({1, -1}));
}

TEST(ConvertersTest,
     CannotConvertTfTensorSpecWithUnsupportedDataTypeToAggTensorSpec) {
  EXPECT_THAT(
      ToAggTensorSpec(CreateTfTensorSpec("foo", tf::DT_VARIANT, {1, 2, 3})),
      StatusIs(INVALID_ARGUMENT));
}

TEST(ConvertersTest, ConvertsNumericTfTensorToAggTensor) {
  tf::TensorProto tensor_proto = PARSE_TEXT_PROTO(R"pb(
    dtype: DT_FLOAT
    tensor_shape {
      dim { size: 2 }
      dim { size: 3 }
    }
    float_val: 1
    float_val: 2
    float_val: 3
    float_val: 4
    float_val: 5
    float_val: 6
  )pb");
  auto tensor = std::make_unique<tf::Tensor>();
  ASSERT_TRUE(tensor->FromProto(tensor_proto));
  EXPECT_THAT(*ToAggTensor(std::move(tensor)),
              IsTensor<float>({2, 3}, {1, 2, 3, 4, 5, 6}));
  EXPECT_THAT(*ToAggTensor(tensor_proto),
              IsTensor<float>({2, 3}, {1, 2, 3, 4, 5, 6}));
}

TEST(ConvertersTest, ConvertsTfStringTensorToAggTensor) {
  tf::TensorProto tensor_proto = PARSE_TEXT_PROTO(R"pb(
    dtype: DT_STRING
    tensor_shape { dim { size: 3 } }
    string_val: "abcd"
    string_val: "foobar"
    string_val: "zzzzzzzzzzzzzz"
  )pb");
  auto tensor = std::make_unique<tf::Tensor>();
  ASSERT_TRUE(tensor->FromProto(tensor_proto));
  EXPECT_THAT(*ToAggTensor(std::move(tensor)),
              IsTensor<string_view>({3}, {"abcd", "foobar", "zzzzzzzzzzzzzz"}));
  EXPECT_THAT(*ToAggTensor(tensor_proto),
              IsTensor<string_view>({3}, {"abcd", "foobar", "zzzzzzzzzzzzzz"}));
}

TEST(ConvertersTest, ConvertsScalaarStringTfTensorToAggTensor) {
  tf::TensorProto tensor_proto = PARSE_TEXT_PROTO(R"pb(
    dtype: DT_STRING
    tensor_shape {}
    string_val: "0123456789"
  )pb");
  auto tensor = std::make_unique<tf::Tensor>();
  ASSERT_TRUE(tensor->FromProto(tensor_proto));
  EXPECT_THAT(*ToAggTensor(std::move(tensor)),
              IsTensor<string_view>({}, {"0123456789"}));
  EXPECT_THAT(*ToAggTensor(tensor_proto),
              IsTensor<string_view>({}, {"0123456789"}));
}

TEST(ConvertersTest, CannotConvertTfTensorWithUnsupportedDataTypeToAggTensor) {
  auto tensor = std::make_unique<tf::Tensor>(tf::DT_VARIANT, CreateTfShape({}));
  EXPECT_THAT(ToAggTensor(std::move(tensor)), StatusIs(INVALID_ARGUMENT));
}

TEST(ConvertersTest, ConvertsAggDataTypeToTfDataType) {
  EXPECT_EQ(*ToTfDataType(DT_FLOAT), tf::DT_FLOAT);
  EXPECT_EQ(*ToTfDataType(DT_DOUBLE), tf::DT_DOUBLE);
  EXPECT_EQ(*ToTfDataType(DT_INT32), tf::DT_INT32);
  EXPECT_EQ(*ToTfDataType(DT_INT64), tf::DT_INT64);
  EXPECT_EQ(*ToTfDataType(DT_STRING), tf::DT_STRING);
}

TEST(ConvertersTest, CannotConvertUnsupportedAggDataTypeToTfDataType) {
  absl::StatusOr<tf::DataType> invalid = ToTfDataType(DT_INVALID);
  EXPECT_FALSE(invalid.ok());
}

TEST(ConvertersTest, ConvertsAggShapeToTfShape) {
  absl::StatusOr<tf::TensorShape> empty_tf_shape = ToTfShape(TensorShape({}));
  ASSERT_OK(empty_tf_shape);
  EXPECT_EQ(*empty_tf_shape, CreateTfShape({}));

  absl::StatusOr<tf::TensorShape> vec_tf_shape = ToTfShape(TensorShape({1}));
  ASSERT_OK(vec_tf_shape);
  EXPECT_EQ(*vec_tf_shape, CreateTfShape({1}));

  absl::StatusOr<tf::TensorShape> matrix_tf_shape =
      ToTfShape(TensorShape({2, 3}));
  ASSERT_OK(matrix_tf_shape);
  EXPECT_EQ(*matrix_tf_shape, CreateTfShape({2, 3}));
}

TEST(ConvertersTest, ConvertsNumericAggTensorToTfTensor) {
  auto tensor = Tensor::Create(DT_FLOAT, {2, 3},
                               CreateTestData<float>({1, 2, 3, 4, 5, 6}));
  absl::StatusOr<tf::Tensor> tf_tensor = ToTfTensor(std::move(*tensor));
  ASSERT_OK(tf_tensor);
  EXPECT_EQ(tf::DT_FLOAT, tf_tensor->dtype());
  EXPECT_EQ(tf::TensorShape({2, 3}), tf_tensor->shape());
  auto flat = tf_tensor->unaligned_flat<float>();
  EXPECT_EQ(flat(0), 1);
  EXPECT_EQ(flat(1), 2);
  EXPECT_EQ(flat(2), 3);
  EXPECT_EQ(flat(3), 4);
  EXPECT_EQ(flat(4), 5);
  EXPECT_EQ(flat(5), 6);
}

TEST(ConvertersTest, ConvertsAggStringTensorToTfTensor) {
  auto tensor = Tensor::Create(
      DT_STRING, {3},
      std::make_unique<VectorStringData>(
          std::vector<std::string>({"abcd", "whimsy", "zzzzz"})));
  absl::StatusOr<tf::Tensor> tf_tensor = ToTfTensor(std::move(*tensor));
  ASSERT_OK(tf_tensor);
  EXPECT_EQ(tf::DT_STRING, tf_tensor->dtype());
  EXPECT_EQ(tf::TensorShape({3}), tf_tensor->shape());
  auto flat = tf_tensor->flat<tf::tstring>();
  EXPECT_EQ(flat(0), "abcd");
  EXPECT_EQ(flat(1), "whimsy");
  EXPECT_EQ(flat(2), "zzzzz");
}

TEST(ConvertersTest, ConvertsAggScalartStringTensorToTfTensor) {
  auto tensor = Tensor::Create(DT_STRING, {},
                               CreateTestData<absl::string_view>({"whimsy"}));
  absl::StatusOr<tf::Tensor> tf_tensor = ToTfTensor(std::move(*tensor));
  ASSERT_OK(tf_tensor);
  EXPECT_EQ(tf::DT_STRING, tf_tensor->dtype());
  EXPECT_EQ(tf::TensorShape({}), tf_tensor->shape());
  auto flat = tf_tensor->flat<tf::tstring>();
  EXPECT_EQ(flat(0), "whimsy");
}

}  // namespace
}  // namespace tensorflow_federated::aggregation::tensorflow

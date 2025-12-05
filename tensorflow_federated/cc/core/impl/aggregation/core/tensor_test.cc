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

#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"

#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/strings/string_view.h"
#include "tensorflow_federated/cc/core/impl/aggregation/base/monitoring.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/datatype.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.pb.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/test_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/testing/testing.h"
#include "tensorflow_federated/cc/testing/protobuf_matchers.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {
namespace aggregation {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

TEST(TensorTest, ConstructNumericScalar) {
  Tensor t(1);
  EXPECT_THAT(t.dtype(), Eq(DT_INT32));
  EXPECT_THAT(t.shape(), Eq(TensorShape{}));
  EXPECT_THAT(t.num_elements(), Eq(1));
  EXPECT_TRUE(t.is_scalar());
  EXPECT_THAT(t.name(), IsEmpty());
  EXPECT_THAT(t, IsTensor<int>({}, {1}));
}

TEST(TensorTest, ConstructStringScalarLiteral) {
  Tensor t("foo");
  EXPECT_THAT(t.dtype(), Eq(DT_STRING));
  EXPECT_THAT(t.num_elements(), Eq(1));
  EXPECT_TRUE(t.is_scalar());
  EXPECT_THAT(t, IsTensor<string_view>({}, {"foo"}));
}

TEST(TensorTest, ConstructStringScalarStringView) {
  absl::string_view s = "foo";
  Tensor t(s);
  EXPECT_THAT(t.dtype(), Eq(DT_STRING));
  EXPECT_THAT(t.num_elements(), Eq(1));
  EXPECT_TRUE(t.is_scalar());
  EXPECT_THAT(t, IsTensor<string_view>({}, {"foo"}));
}

TEST(TensorTest, ConstructStringScalarStringByConstRef) {
  std::string s = "foo";
  const auto& ref = s;
  Tensor t(ref);
  EXPECT_THAT(t.dtype(), Eq(DT_STRING));
  EXPECT_THAT(t.num_elements(), Eq(1));
  EXPECT_TRUE(t.is_scalar());
  EXPECT_THAT(t, IsTensor<string_view>({}, {"foo"}));
}

TEST(TensorTest, ConstructStringScalarStringByCopy) {
  std::string s = "foo";
  Tensor t(s);
  EXPECT_THAT(t.dtype(), Eq(DT_STRING));
  EXPECT_THAT(t.num_elements(), Eq(1));
  EXPECT_TRUE(t.is_scalar());
  EXPECT_THAT(t, IsTensor<string_view>({}, {"foo"}));
}

TEST(TensorTest, ConstructStringScalarStringByMove) {
  std::string s = "foo";
  Tensor t(std::move(s));
  EXPECT_THAT(t.dtype(), Eq(DT_STRING));
  EXPECT_THAT(t.num_elements(), Eq(1));
  EXPECT_TRUE(t.is_scalar());
  EXPECT_THAT(t, IsTensor<string_view>({}, {"foo"}));
}

TEST(TensorTest, ConstructNumericVector) {
  Tensor t({1, 2, 3});
  EXPECT_THAT(t.dtype(), Eq(DT_INT32));
  EXPECT_THAT(t.num_elements(), Eq(3));
  EXPECT_FALSE(t.is_scalar());
  EXPECT_THAT(t, IsTensor<int>({3}, {1, 2, 3}));
}

TEST(TensorTest, ConstructStringVector) {
  Tensor t({"foo", "bar"});
  EXPECT_THAT(t.dtype(), Eq(DT_STRING));
  EXPECT_THAT(t.num_elements(), Eq(2));
  EXPECT_FALSE(t.is_scalar());
  EXPECT_THAT(t, IsTensor<string_view>({2}, {"foo", "bar"}));
}

TEST(TensorTest, ConstructNumericScalarWithName) {
  Tensor t(1, "test_name");
  EXPECT_THAT(t.name(), Eq("test_name"));
  EXPECT_THAT(t, IsTensor<int>({}, {1}));
}

TEST(TensorTest, ConstructStringScalarWithName) {
  Tensor t("abc", "test_name");
  EXPECT_THAT(t.name(), Eq("test_name"));
  EXPECT_THAT(t, IsTensor<string_view>({}, {"abc"}));
}

TEST(TensorTest, ConstructNumericVectorWithName) {
  Tensor t({1, 2, 3}, "test_name");
  EXPECT_THAT(t.name(), Eq("test_name"));
  EXPECT_THAT(t, IsTensor<int>({3}, {1, 2, 3}));
}

TEST(TensorTest, ConstructStringVectorWithName) {
  Tensor t({"foo", "bar"}, "test_name");
  EXPECT_THAT(t.name(), Eq("test_name"));
  EXPECT_THAT(t, IsTensor<string_view>({2}, {"foo", "bar"}));
}

TEST(TensorTest, CreateDense) {
  auto t =
      Tensor::Create(DT_FLOAT, {2, 2}, CreateTestData<float>({1, 2, 3, 4}));
  EXPECT_THAT(t, IsOk());
  EXPECT_THAT(t->dtype(), Eq(DT_FLOAT));
  EXPECT_THAT(t->num_elements(), Eq(4));
  EXPECT_TRUE(t->is_dense());
  EXPECT_THAT(*t, IsTensor({2, 2}, {1.f, 2.f, 3.f, 4.f}));
}

TEST(TensorTest, CreateZeroDataSize) {
  auto t = Tensor::Create(DT_INT32, {0}, CreateTestData<int>({}));
  EXPECT_THAT(t, IsOk());
  EXPECT_THAT(t->dtype(), Eq(DT_INT32));
  EXPECT_THAT(t->shape(), Eq(TensorShape{0}));
  EXPECT_THAT(t->num_elements(), Eq(0));
  EXPECT_TRUE(t->is_dense());
  EXPECT_THAT(t->AsAggVector<int>().size(), Eq(0));
}

TEST(TensorTest, TensorCreateWithNameSuccess) {
  TFF_ASSERT_OK_AND_ASSIGN(
      Tensor t,
      Tensor::Create(/*dtype=*/DT_INT32, /*shape=*/{1},
                     /*data=*/CreateTestData<int>({1}), /*name=*/"test_name"));
  EXPECT_EQ(t.name(), "test_name");
}

TEST(TensorTest, CreateShapeWithUnknownDimensions) {
  auto t = Tensor::Create(DT_FLOAT, {-1}, CreateTestData<float>({1, 2, 3}));
  EXPECT_THAT(t, StatusIs(INVALID_ARGUMENT));
}

TEST(TensorTest, CreateDataValidationError) {
  auto t = Tensor::Create(DT_FLOAT, {}, CreateTestData<char>({'a', 'b', 'c'}));
  EXPECT_THAT(t, StatusIs(FAILED_PRECONDITION));
}

TEST(TensorTest, CreateDataSizeError) {
  auto t = Tensor::Create(DT_FLOAT, {1}, CreateTestData<float>({1, 2}));
  EXPECT_THAT(t, StatusIs(FAILED_PRECONDITION));
}

struct FooBar {};

TEST(TensorTest, AsAggVectorTypeCheckFailure) {
  Tensor t({1.f});
  EXPECT_DEATH(t.AsAggVector<FooBar>(), "Incompatible tensor dtype()");
  EXPECT_DEATH(t.AsAggVector<int>(), "Incompatible tensor dtype()");
}

TEST(TensorTest, CastToScalarIntScalarTensor) {
  Tensor t(10);
  EXPECT_THAT(t.CastToScalar<float>(), ::testing::FloatNear(10, 1e-5f));
  EXPECT_THAT(t.CastToScalar<double>(), ::testing::DoubleNear(10, 1e-5));
  EXPECT_EQ(t.CastToScalar<int>(), 10);
}

TEST(TensorTest, CastToScalarFloatScalarTensor) {
  Tensor t(5.3f);
  EXPECT_THAT(t.CastToScalar<float>(), ::testing::FloatNear(5.3f, 1e-5f));
  EXPECT_THAT(t.CastToScalar<double>(), ::testing::DoubleNear(5.3, 1e-5));
  EXPECT_EQ(t.CastToScalar<int>(), 5);
}

TEST(TensorTest, CastToScalarNumericalScalarTensorWithRounding) {
  Tensor t1(2.9999f);
  EXPECT_EQ(t1.CastToScalar<int>(), 3);

  Tensor t2(3.0001f);
  EXPECT_EQ(t2.CastToScalar<int>(), 3);

  Tensor t3(-2.9999);
  EXPECT_EQ(t3.CastToScalar<int>(), -3);

  Tensor t4(-3.0001);
  EXPECT_EQ(t4.CastToScalar<int>(), -3);
}

TEST(TensorTest, CastToScalarStringScalarTensor) {
  Tensor t("foo");
  EXPECT_EQ(t.CastToScalar<string_view>(), "foo");
}

TEST(TensorTest, CastToScalarMismatchType) {
  Tensor t1("foo");
  EXPECT_DEATH(t1.CastToScalar<int>(), "Unsupported type");

  Tensor t2(5.5f);
  EXPECT_DEATH(t2.CastToScalar<string_view>(), "Incompatible tensor dtype()");
}

TEST(TensorTest, CastToScalarNonScalar) {
  Tensor t1({"foo", "bar"});
  EXPECT_DEATH(t1.CastToScalar<string_view>(),
               "CastToScalar should only be used on scalar tensors");

  Tensor t2({5.5f, 5.7f, 5.9f});
  EXPECT_DEATH(t2.CastToScalar<float>(),
               "CastToScalar should only be used on scalar tensors");
}

TEST(TensorTest, AsScalarIntScalarTensor) {
  Tensor t(10);
  EXPECT_EQ(t.AsScalar<int>(), 10);
}

TEST(TensorTest, AsScalarFloatScalarTensor) {
  Tensor t(5.3f);
  EXPECT_THAT(t.AsScalar<float>(), ::testing::FloatNear(5.3f, 1e-5f));
}

TEST(TensorTest, AsScalarStringScalarTensor) {
  Tensor t("foo");
  EXPECT_EQ(t.AsScalar<string_view>(), "foo");
}

TEST(TensorTest, AsScalarMismatchType) {
  Tensor t1("foo");
  EXPECT_DEATH(t1.AsScalar<int>(), "Incompatible tensor dtype()");

  Tensor t2(5.5f);
  EXPECT_DEATH(t2.AsScalar<string_view>(), "Incompatible tensor dtype()");
}

TEST(TensorTest, AsScalarNonScalar) {
  Tensor t1({"foo", "bar"});
  EXPECT_DEATH(t1.AsScalar<string_view>(),
               "AsScalar should only be used on scalar tensors");

  Tensor t2({5.5f, 5.7f, 5.9f});
  EXPECT_DEATH(t2.AsScalar<float>(),
               "AsScalar should only be used on scalar tensors");
}

TEST(TensorTest, AsSpanNumericTensor) {
  Tensor t({5.5f, 5.7f, 5.9f});
  auto span = t.AsSpan<float>();
  EXPECT_EQ(span.size(), 3);
  EXPECT_EQ(span.at(0), 5.5f);
  EXPECT_EQ(span.at(1), 5.7f);
  EXPECT_EQ(span.at(2), 5.9f);
}

TEST(TensorTest, AsSpanStringTensor) {
  Tensor t({"foo", "bar"});
  auto span = t.AsSpan<string_view>();
  EXPECT_EQ(span.size(), 2);
  EXPECT_EQ(span.at(0), "foo");
  EXPECT_EQ(span.at(1), "bar");
}

TEST(TensorTest, AsSpanMismatchType) {
  Tensor t({"foo", "bar"});
  EXPECT_DEATH(t.AsSpan<int>(), "Incompatible tensor dtype()");
}

TEST(TensorTest, ToProtoInt32Success) {
  std::initializer_list<int32_t> values{1, 2, 3, 4};
  auto t = Tensor::Create(DT_INT32, {2, 2}, CreateTestData(values));
  TensorProto expected_proto;
  expected_proto.set_dtype(DT_INT32);
  expected_proto.mutable_shape()->add_dim_sizes(2);
  expected_proto.mutable_shape()->add_dim_sizes(2);
  expected_proto.set_content(ToProtoContent(values));
  EXPECT_THAT(t->ToProto(), testing::EqualsProto(expected_proto));
}

TEST(TensorTest, ToProtoUint64Success) {
  std::initializer_list<uint64_t> values{4294967296, 4294967297, 4294967298,
                                         4294967299};
  auto t = Tensor::Create(DT_UINT64, {2, 2}, CreateTestData(values));
  TensorProto expected_proto;
  expected_proto.set_dtype(DT_UINT64);
  expected_proto.mutable_shape()->add_dim_sizes(2);
  expected_proto.mutable_shape()->add_dim_sizes(2);
  expected_proto.set_content(ToProtoContent(values));
  EXPECT_THAT(t->ToProto(), testing::EqualsProto(expected_proto));
}

TEST(TensorTest, ToProtoStringSuccess) {
  std::initializer_list<string_view> values{"abc",  "de",    "",
                                            "fghi", "jklmn", "o"};
  auto t = Tensor::Create(DT_STRING, {2, 3}, CreateTestData(values));
  TensorProto expected_proto;
  expected_proto.set_dtype(DT_STRING);
  expected_proto.mutable_shape()->add_dim_sizes(2);
  expected_proto.mutable_shape()->add_dim_sizes(3);
  expected_proto.set_content(ToProtoContent(values));
  EXPECT_THAT(t->ToProto(), testing::EqualsProto(expected_proto));
}

TEST(TensorTest, ToProtoWithNameSuccess) {
  std::initializer_list<int32_t> values{1};
  auto t = Tensor::Create(DT_INT32, {1}, CreateTestData(values), "test_name");
  TensorProto expected_proto;
  expected_proto.set_dtype(DT_INT32);
  expected_proto.mutable_shape()->add_dim_sizes(1);
  expected_proto.set_content(ToProtoContent(values));
  expected_proto.set_name("test_name");
  EXPECT_THAT(t->ToProto(), testing::EqualsProto(expected_proto));
}

TEST(TensorTest, FromProtoInt32Success) {
  std::initializer_list<int32_t> values{5, 6, 7, 8, 9, 10};
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_INT32);
  tensor_proto.mutable_shape()->add_dim_sizes(2);
  tensor_proto.mutable_shape()->add_dim_sizes(3);
  tensor_proto.set_content(ToProtoContent(values));
  auto t = Tensor::FromProto(tensor_proto);
  EXPECT_THAT(t, IsOk());
  EXPECT_THAT(*t, IsTensor({2, 3}, values));
}

TEST(TensorTest, FromProtoUint64Success) {
  std::initializer_list<uint64_t> values{4294967296, 4294967297, 4294967298,
                                         4294967299, 4294967300, 4294967301};
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_UINT64);
  tensor_proto.mutable_shape()->add_dim_sizes(2);
  tensor_proto.mutable_shape()->add_dim_sizes(3);
  tensor_proto.set_content(ToProtoContent(values));
  auto t = Tensor::FromProto(tensor_proto);
  EXPECT_THAT(t, IsOk());
  EXPECT_THAT(*t, IsTensor({2, 3}, values));
}

TEST(TensorTest, FromProtoStringSuccess) {
  std::initializer_list<string_view> values{"aaaaaaaa", "b", "cccc", "ddddddd"};
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_STRING);
  tensor_proto.mutable_shape()->add_dim_sizes(2);
  tensor_proto.mutable_shape()->add_dim_sizes(2);
  tensor_proto.set_content(ToProtoContent(values));
  auto t = Tensor::FromProto(tensor_proto);
  EXPECT_THAT(t, IsOk());
  EXPECT_THAT(*t, IsTensor({2, 2}, values));
}

TEST(TensorTest, FromProtoInt32WithoutContentSuccess) {
  std::initializer_list<int32_t> values{5, 6, 7, 8, 9, 10};
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_INT32);
  tensor_proto.mutable_shape()->add_dim_sizes(2);
  tensor_proto.mutable_shape()->add_dim_sizes(3);
  for (int32_t v : values) {
    tensor_proto.add_int_val(v);
  }
  auto t = Tensor::FromProto(tensor_proto);
  EXPECT_THAT(t, IsOk());
  EXPECT_THAT(*t, IsTensor({2, 3}, values));
}

TEST(TensorTest, FromProtoFloatWithoutContentSuccess) {
  std::initializer_list<float> values{1.2, 1.4, 1.6};
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_FLOAT);
  tensor_proto.mutable_shape()->add_dim_sizes(3);
  for (auto v : values) {
    tensor_proto.add_float_val(v);
  }
  auto t = Tensor::FromProto(tensor_proto);
  EXPECT_THAT(t, IsOk());
  EXPECT_THAT(*t, IsTensor({3}, values));
}

TEST(TensorTest, FromProtoStringWithoutContentSuccess) {
  std::initializer_list<string_view> values{"a", "b", "c", "d"};
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_STRING);
  tensor_proto.mutable_shape()->add_dim_sizes(4);
  for (auto v : values) {
    tensor_proto.add_string_val(std::string(v));
  }
  auto t = Tensor::FromProto(tensor_proto);
  EXPECT_THAT(t, IsOkAndHolds(IsTensor({4}, values)));
}

TEST(TensorTest, FromProtoWithNameSuccess) {
  std::initializer_list<int32_t> values{1};
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_INT32);
  tensor_proto.mutable_shape()->add_dim_sizes(1);
  tensor_proto.set_content(ToProtoContent(values));
  tensor_proto.set_name("test_name");
  auto t = Tensor::FromProto(tensor_proto);
  EXPECT_THAT(t, IsOkAndHolds(IsTensor({1}, values)));
  EXPECT_EQ(t->name(), "test_name");
}

TEST(TensorTest, LargeStringValuesSerialization) {
  std::string s1(123456, 'a');
  std::string s2(7890, 'b');
  std::string s3(1357924, 'c');
  auto t1 =
      Tensor::Create(DT_STRING, {3}, CreateTestData<string_view>({s1, s2, s3}));
  auto proto = t1->ToProto();
  auto t2 = Tensor::FromProto(proto);
  EXPECT_THAT(*t2, IsTensor<string_view>({3}, {s1, s2, s3}));
}

TEST(TensorTest, FromProtoMutableSuccess) {
  std::initializer_list<int32_t> values{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_INT32);
  tensor_proto.mutable_shape()->add_dim_sizes(10);
  tensor_proto.set_content(ToProtoContent(values));
  // Store the data pointer to make sure that the tensor retains the same data.
  void* data_ptr = tensor_proto.mutable_content()->data();
  auto t = Tensor::FromProto(std::move(tensor_proto));
  EXPECT_THAT(t, IsOk());
  EXPECT_THAT(*t, IsTensor({10}, values));
  EXPECT_EQ(data_ptr, t->data().data());
}

TEST(TensorTest, FromProtoNegativeDimSize) {
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_INT32);
  tensor_proto.mutable_shape()->add_dim_sizes(-1);
  tensor_proto.set_content(ToProtoContent<int32_t>({1}));
  EXPECT_THAT(Tensor::FromProto(tensor_proto), StatusIs(INVALID_ARGUMENT));
}

TEST(TensorTest, FromProtoMultipleFields) {
  std::initializer_list<int32_t> values{5, 6, 7, 8, 9, 10};
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_INT32);
  tensor_proto.mutable_shape()->add_dim_sizes(2);
  tensor_proto.mutable_shape()->add_dim_sizes(3);
  tensor_proto.set_content(ToProtoContent(values));
  for (int32_t v : values) {
    tensor_proto.add_int_val(v);
  }
  Status s = Tensor::FromProto(tensor_proto).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(
      s.message(),
      HasSubstr("Tensor proto contains multiple representations of data."));
}

TEST(TensorTest, FromProtoMismatchedType) {
  std::initializer_list<int32_t> values{5, 6, 7, 8, 9, 10};
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_FLOAT);
  tensor_proto.mutable_shape()->add_dim_sizes(2);
  tensor_proto.mutable_shape()->add_dim_sizes(3);
  for (int32_t v : values) {
    tensor_proto.add_int_val(v);
  }
  Status s = Tensor::FromProto(tensor_proto).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(),
              HasSubstr("Tensor proto contains data of unexpected data type"));
}

TEST(TensorTest, FromProtoNoData) {
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_STRING);
  tensor_proto.mutable_shape()->add_dim_sizes(0);
  auto t = Tensor::FromProto(tensor_proto);
  EXPECT_THAT(t, IsOk());
  EXPECT_THAT(t->shape().dim_sizes().size(), 1);
  EXPECT_THAT(t->shape().dim_sizes()[0], 0);
}

TEST(TensorTest, FromProtoNoDataMismatchShape) {
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_STRING);
  tensor_proto.mutable_shape()->add_dim_sizes(2);
  Status s = Tensor::FromProto(tensor_proto).status();
  EXPECT_THAT(s, StatusIs(INVALID_ARGUMENT));
  EXPECT_THAT(s.message(), HasSubstr("Tensor proto contains no data but the "
                                     "shape indicates it is non-empty"));
}

TEST(TensorTest, FromProtoInvalidStringContent) {
  TensorProto tensor_proto;
  tensor_proto.set_dtype(DT_STRING);
  tensor_proto.mutable_shape()->add_dim_sizes(1);

  std::string content(1, '\5');
  tensor_proto.set_content(content);
  EXPECT_THAT(Tensor::FromProto(tensor_proto), StatusIs(INVALID_ARGUMENT));

  content.append("abc");
  tensor_proto.set_content(content);
  EXPECT_THAT(Tensor::FromProto(tensor_proto), StatusIs(INVALID_ARGUMENT));
}

TEST(TensorTest, RoundTripDataInt) {
  std::initializer_list<int32_t> values{1, 2, 3, 4};
  auto t = Tensor::Create(DT_INT32, {2, 2}, CreateTestData(values));

  auto p = t->ToProto();
  EXPECT_THAT(p.shape().dim_sizes_size(), 2);
  EXPECT_THAT(p.shape().dim_sizes(0), 2);
  EXPECT_THAT(p.shape().dim_sizes(1), 2);

  auto result = Tensor::FromProto(p);
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(*result, IsTensor({2, 2}, values));
}

TEST(TensorTest, RoundTripDataString) {
  std::initializer_list<string_view> values{"abc",  "de",    "",
                                            "fghi", "jklmn", "o"};
  auto t = Tensor::Create(DT_STRING, {2, 3}, CreateTestData(values));

  auto p = t->ToProto();
  EXPECT_THAT(p.shape().dim_sizes_size(), 2);
  EXPECT_THAT(p.shape().dim_sizes(0), 2);
  EXPECT_THAT(p.shape().dim_sizes(1), 3);

  auto result = Tensor::FromProto(p);
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(*result, IsTensor({2, 3}, values));
}

TEST(TensorTest, RoundTripNoDataInt) {
  std::initializer_list<int32_t> values{};
  auto t = Tensor::Create(DT_INT32, {0}, CreateTestData(values));

  auto p = t->ToProto();
  EXPECT_THAT(p.shape().dim_sizes_size(), 1);
  EXPECT_THAT(p.shape().dim_sizes(0), 0);

  auto result = Tensor::FromProto(p);
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(*result, IsTensor({0}, values));
}

TEST(TensorTest, RoundTripNoDataString) {
  std::initializer_list<string_view> values{};
  auto t = Tensor::Create(DT_STRING, {0}, CreateTestData(values));

  auto p = t->ToProto();
  EXPECT_THAT(p.shape().dim_sizes_size(), 1);
  EXPECT_THAT(p.shape().dim_sizes(0), 0);

  auto result = Tensor::FromProto(p);
  EXPECT_THAT(result, IsOk());
  EXPECT_THAT(*result, IsTensor({0}, values));
}

TEST(TensorTest, SetNameSuccess) {
  TFF_ASSERT_OK_AND_ASSIGN(
      Tensor tensor,
      Tensor::Create(DT_FLOAT, {}, CreateTestData<float>({1.0f})));
  // The tensor name should be empty initially.
  EXPECT_THAT(tensor.name(), IsEmpty());

  // Set the tensor name.
  TFF_EXPECT_OK(tensor.set_name("my_test_tensor"));

  // Verify the name has been set correctly.
  EXPECT_THAT(tensor.name(), Eq("my_test_tensor"));
}

}  // namespace
}  // namespace aggregation
}  // namespace tensorflow_federated

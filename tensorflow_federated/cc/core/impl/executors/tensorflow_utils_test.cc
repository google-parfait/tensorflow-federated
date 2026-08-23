/* Copyright 2024, The TensorFlow Federated Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License
==============================================================================*/

#include "tensorflow_federated/cc/core/impl/executors/tensorflow_utils.h"

#include <complex>
#include <cstdint>
#include <string>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/cord_test_helpers.h"
#include "absl/strings/string_view.h"
#include "Eigen/Core"
#include "federated_language/proto/array.pb.h"
#include "federated_language/proto/computation.pb.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow_federated/cc/core/impl/executors/array_shape_test_utils.h"
#include "tensorflow_federated/cc/core/impl/executors/array_test_utils.h"
#include "tensorflow_federated/cc/testing/protobuf_matchers.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {
namespace {

TEST(TensorShapeFromArrayShapeTest, TestReturnsTensorShape_fully_defined) {
  const federated_language::ArrayShape& shape_pb =
      testing::CreateArrayShape({2, 3});
  const tensorflow::TensorShape& expected_shape =
      tensorflow::TensorShape({2, 3});

  const tensorflow::TensorShape& actual_shape =
      TFF_ASSERT_OK(TensorShapeFromArrayShape(shape_pb));

  EXPECT_EQ(actual_shape, expected_shape);
}

TEST(TensorShapeFromArrayShapeTest, TestReturnsTensorShape_scalar) {
  const federated_language::ArrayShape& shape_pb =
      testing::CreateArrayShape({});
  const tensorflow::TensorShape& expected_shape = tensorflow::TensorShape({});

  const tensorflow::TensorShape& actual_shape =
      TFF_ASSERT_OK(TensorShapeFromArrayShape(shape_pb));

  EXPECT_EQ(actual_shape, expected_shape);
}

TEST(TensorShapeFromArrayShapeTest, TestFails_partially_defined) {
  const federated_language::ArrayShape& shape_pb =
      testing::CreateArrayShape({2, -1});

  const absl::StatusOr<tensorflow::TensorShape>& result =
      TensorShapeFromArrayShape(shape_pb);

  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(TensorShapeFromArrayShapeTest, TestFails_unknown) {
  const federated_language::ArrayShape& shape_pb =
      testing::CreateArrayShape({}, true);

  const absl::StatusOr<tensorflow::TensorShape>& result =
      TensorShapeFromArrayShape(shape_pb);

  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

struct PartialTensorShapeFromArrayShapeTestCase {
  std::string test_name;
  const federated_language::ArrayShape shape_pb;
  const tensorflow::PartialTensorShape expected_shape;
};

using PartialTensorShapeFromArrayShapeTest =
    ::testing::TestWithParam<PartialTensorShapeFromArrayShapeTestCase>;

TEST_P(PartialTensorShapeFromArrayShapeTest, TestReturnsPartialTensorShape) {
  const PartialTensorShapeFromArrayShapeTestCase& test_case = GetParam();

  const tensorflow::PartialTensorShape& actual_shape =
      PartialTensorShapeFromArrayShape(test_case.shape_pb);

  EXPECT_TRUE(actual_shape.IsIdenticalTo(test_case.expected_shape));
}

INSTANTIATE_TEST_SUITE_P(
    PartialTensorShapeFromArrayShapeTestSuiteInstantiation,
    PartialTensorShapeFromArrayShapeTest,
    ::testing::ValuesIn<PartialTensorShapeFromArrayShapeTestCase>({
        {
            "fully_defined",
            testing::CreateArrayShape({2, 3}),
            tensorflow::PartialTensorShape({2, 3}),
        },
        {
            "partially_defined",
            testing::CreateArrayShape({2, -1}),
            tensorflow::PartialTensorShape({2, -1}),
        },
        {
            "unknown",
            testing::CreateArrayShape({}, true),
            tensorflow::PartialTensorShape(),
        },
        {
            "scalar",
            testing::CreateArrayShape({}),
            tensorflow::PartialTensorShape({}),
        },
    }),
    [](const ::testing::TestParamInfo<
        PartialTensorShapeFromArrayShapeTest::ParamType>& info) {
      return info.param.test_name;
    });

struct ArrayFromTensorTestCase {
  std::string test_name;
  const tensorflow::Tensor tensor;
  const federated_language::Array expected_array_pb;
};

using ArrayFromTensorTest = ::testing::TestWithParam<ArrayFromTensorTestCase>;

TEST_P(ArrayFromTensorTest, TestReturnsTensor) {
  const ArrayFromTensorTestCase& test_case = GetParam();

  const federated_language::Array& actual_array_pb =
      TFF_ASSERT_OK(ArrayFromTensor(test_case.tensor));

  EXPECT_THAT(actual_array_pb,
              testing::EqualsProto(test_case.expected_array_pb));
}

INSTANTIATE_TEST_SUITE_P(
    ArrayFromTensorTestSuiteInstantiation, ArrayFromTensorTest,
    ::testing::ValuesIn<ArrayFromTensorTestCase>({
        {
            "bool",
            tensorflow::test::AsScalar(true),
            testing::CreateArrayOrDie(federated_language::DataType::DT_BOOL,
                                      testing::CreateArrayShape({}), {true}),
        },
        {
            "int8",
            tensorflow::test::AsScalar<int8_t>(1),
            testing::CreateArrayOrDie(federated_language::DataType::DT_INT8,
                                      testing::CreateArrayShape({}), {1}),
        },
        {
            "int16",
            tensorflow::test::AsScalar<int16_t>(1),
            testing::CreateArrayOrDie(federated_language::DataType::DT_INT16,
                                      testing::CreateArrayShape({}), {1}),
        },
        {
            "int32",
            tensorflow::test::AsScalar<int32_t>(1),
            testing::CreateArrayOrDie(federated_language::DataType::DT_INT32,
                                      testing::CreateArrayShape({}), {1}),
        },
        {
            "int64",
            tensorflow::test::AsScalar<int64_t>(1),
            testing::CreateArrayOrDie(federated_language::DataType::DT_INT64,
                                      testing::CreateArrayShape({}), {1}),
        },
        {
            "uint8",
            tensorflow::test::AsScalar<uint8_t>(1),
            testing::CreateArrayOrDie(federated_language::DataType::DT_UINT8,
                                      testing::CreateArrayShape({}), {1}),
        },
        {
            "uint16",
            tensorflow::test::AsScalar<uint16_t>(1),
            testing::CreateArrayOrDie(federated_language::DataType::DT_UINT16,
                                      testing::CreateArrayShape({}), {1}),
        },
        {
            "uint32",
            tensorflow::test::AsScalar<uint32_t>(1),
            testing::CreateArrayOrDie(federated_language::DataType::DT_UINT32,
                                      testing::CreateArrayShape({}), {1}),
        },
        {
            "uint64",
            tensorflow::test::AsScalar<uint64_t>(1),
            testing::CreateArrayOrDie(federated_language::DataType::DT_UINT64,
                                      testing::CreateArrayShape({}), {1}),
        },
        {
            "float16",
            tensorflow::test::AsScalar(Eigen::half{1.0}),
            testing::CreateArrayOrDie(federated_language::DataType::DT_HALF,
                                      testing::CreateArrayShape({}),
                                      {Eigen::half{1.0}}),
        },
        {
            "float32",
            tensorflow::test::AsScalar<float>(1.0),
            testing::CreateArrayOrDie(federated_language::DataType::DT_FLOAT,
                                      testing::CreateArrayShape({}), {1.0}),
        },
        {
            "float64",
            tensorflow::test::AsScalar<double>(1.0),
            testing::CreateArrayOrDie(federated_language::DataType::DT_DOUBLE,
                                      testing::CreateArrayShape({}), {1.0}),
        },
        {
            "complex64",
            tensorflow::test::AsScalar(tensorflow::complex64{1.0, 1.0}),
            testing::CreateArrayOrDie(
                federated_language::DataType::DT_COMPLEX64,
                testing::CreateArrayShape({}), {std::complex<float>(1.0, 1.0)}),
        },
        {
            "complex128",
            tensorflow::test::AsScalar(tensorflow::complex128{1.0, 1.0}),
            testing::CreateArrayOrDie(
                federated_language::DataType::DT_COMPLEX128,
                testing::CreateArrayShape({}),
                {std::complex<double>(1.0, 1.0)}),
        },
        {
            "bfloat16",
            tensorflow::test::AsScalar(Eigen::bfloat16{1.0}),
            testing::CreateArrayOrDie(federated_language::DataType::DT_BFLOAT16,
                                      testing::CreateArrayShape({}),
                                      {Eigen::bfloat16{1.0}}),
        },
        {
            "string",
            tensorflow::test::AsScalar<tensorflow::tstring>("a"),
            testing::CreateArrayOrDie(federated_language::DataType::DT_STRING,
                                      testing::CreateArrayShape({}), {"a"}),
        },
        {
            "array",
            tensorflow::test::AsTensor<int32_t>(
                {1, 2, 3, 4, 5, 6}, tensorflow::TensorShape({2, 3})),
            testing::CreateArrayOrDie(federated_language::DataType::DT_INT32,
                                      testing::CreateArrayShape({2, 3}),
                                      {1, 2, 3, 4, 5, 6}),
        },
    }),
    [](const ::testing::TestParamInfo<ArrayFromTensorTest::ParamType>& info) {
      return info.param.test_name;
    });

struct TensorFromArrayTestCase {
  std::string test_name;
  const federated_language::Array array_pb;
  const tensorflow::Tensor expected_tensor;
};

using TensorFromArrayTest = ::testing::TestWithParam<TensorFromArrayTestCase>;

TEST_P(TensorFromArrayTest, TestReturnsTensor) {
  const TensorFromArrayTestCase& test_case = GetParam();

  const tensorflow::Tensor& actual_tensor =
      TFF_ASSERT_OK(TensorFromArray(test_case.array_pb));

  tensorflow::test::ExpectEqual(actual_tensor, test_case.expected_tensor);
}

INSTANTIATE_TEST_SUITE_P(
    TensorFromArrayTestSuiteInstantiation, TensorFromArrayTest,
    ::testing::ValuesIn<TensorFromArrayTestCase>({
        {
            "bool",
            testing::CreateArrayOrDie(federated_language::DataType::DT_BOOL,
                                      testing::CreateArrayShape({}), {true}),
            tensorflow::test::AsScalar(true),
        },
        {
            "int8",
            testing::CreateArrayOrDie(federated_language::DataType::DT_INT8,
                                      testing::CreateArrayShape({}), {1}),
            tensorflow::test::AsScalar<int8_t>(1),
        },
        {
            "int16",
            testing::CreateArrayOrDie(federated_language::DataType::DT_INT16,
                                      testing::CreateArrayShape({}), {1}),
            tensorflow::test::AsScalar<int16_t>(1),
        },
        {
            "int32",
            testing::CreateArrayOrDie(federated_language::DataType::DT_INT32,
                                      testing::CreateArrayShape({}), {1}),
            tensorflow::test::AsScalar<int32_t>(1),
        },
        {
            "int64",
            testing::CreateArrayOrDie(federated_language::DataType::DT_INT64,
                                      testing::CreateArrayShape({}), {1}),
            tensorflow::test::AsScalar<int64_t>(1),
        },
        {
            "uint8",
            testing::CreateArrayOrDie(federated_language::DataType::DT_UINT8,
                                      testing::CreateArrayShape({}), {1}),
            tensorflow::test::AsScalar<uint8_t>(1),
        },
        {
            "uint16",
            testing::CreateArrayOrDie(federated_language::DataType::DT_UINT16,
                                      testing::CreateArrayShape({}), {1}),
            tensorflow::test::AsScalar<uint16_t>(1),
        },
        {
            "uint32",
            testing::CreateArrayOrDie(federated_language::DataType::DT_UINT32,
                                      testing::CreateArrayShape({}), {1}),
            tensorflow::test::AsScalar<uint32_t>(1),
        },
        {
            "uint64",
            testing::CreateArrayOrDie(federated_language::DataType::DT_UINT64,
                                      testing::CreateArrayShape({}), {1}),
            tensorflow::test::AsScalar<uint64_t>(1),
        },
        {
            "float16",
            testing::CreateArrayOrDie(federated_language::DataType::DT_HALF,
                                      testing::CreateArrayShape({}),
                                      {Eigen::half{1.0}}),
            tensorflow::test::AsScalar(Eigen::half{1.0}),
        },
        {
            "float32",
            testing::CreateArrayOrDie(federated_language::DataType::DT_FLOAT,
                                      testing::CreateArrayShape({}), {1.0}),
            tensorflow::test::AsScalar<float>(1.0),
        },
        {
            "float64",
            testing::CreateArrayOrDie(federated_language::DataType::DT_DOUBLE,
                                      testing::CreateArrayShape({}), {1.0}),
            tensorflow::test::AsScalar<double>(1.0),
        },
        {
            "complex64",
            testing::CreateArrayOrDie(
                federated_language::DataType::DT_COMPLEX64,
                testing::CreateArrayShape({}), {std::complex<float>(1.0, 1.0)}),
            tensorflow::test::AsScalar(tensorflow::complex64{1.0, 1.0}),
        },
        {
            "complex128",
            testing::CreateArrayOrDie(
                federated_language::DataType::DT_COMPLEX128,
                testing::CreateArrayShape({}),
                {std::complex<double>(1.0, 1.0)}),
            tensorflow::test::AsScalar(tensorflow::complex128{1.0, 1.0}),
        },
        {
            "bfloat16",
            testing::CreateArrayOrDie(federated_language::DataType::DT_BFLOAT16,
                                      testing::CreateArrayShape({}),
                                      {Eigen::bfloat16{1.0}}),
            tensorflow::test::AsScalar(Eigen::bfloat16{1.0}),
        },
        {
            "string",
            testing::CreateArrayOrDie(federated_language::DataType::DT_STRING,
                                      testing::CreateArrayShape({}), {"a"}),
            tensorflow::test::AsScalar<tensorflow::tstring>("a"),
        },
        {
            "array",
            testing::CreateArrayOrDie(federated_language::DataType::DT_INT32,
                                      testing::CreateArrayShape({2, 3}),
                                      {1, 2, 3, 4, 5, 6}),
            tensorflow::test::AsTensor<int32_t>(
                {1, 2, 3, 4, 5, 6}, tensorflow::TensorShape({2, 3})),
        },
    }),
    [](const ::testing::TestParamInfo<TensorFromArrayTest::ParamType>& info) {
      return info.param.test_name;
    });

struct ArrayContentFromTensorTestCase {
  std::string test_name;
  const tensorflow::Tensor tensor;
  const federated_language::Array expected_array_pb;
};

using ArrayContentFromTensorTest =
    ::testing::TestWithParam<ArrayContentFromTensorTestCase>;

TEST_P(ArrayContentFromTensorTest, TestReturnsTensor) {
  const ArrayContentFromTensorTestCase& test_case = GetParam();

  const federated_language::Array& actual_array_pb =
      TFF_ASSERT_OK(ArrayContentFromTensor(test_case.tensor));

  EXPECT_THAT(actual_array_pb,
              testing::EqualsProto(test_case.expected_array_pb));
}

#define CONTENT(s) absl::string_view(s, sizeof(s) - 1)

INSTANTIATE_TEST_SUITE_P(
    ArrayContentFromTensorTestSuiteInstantiation, ArrayContentFromTensorTest,
    ::testing::ValuesIn<ArrayContentFromTensorTestCase>({
        {
            "bool",
            tensorflow::test::AsScalar(true),
            testing::CreateArrayContent(federated_language::DataType::DT_BOOL,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001")),
        },
        {
            "int8",
            tensorflow::test::AsScalar<int8_t>(1),
            testing::CreateArrayContent(federated_language::DataType::DT_INT8,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001")),
        },
        {
            "int16",
            tensorflow::test::AsScalar<int16_t>(1),
            testing::CreateArrayContent(federated_language::DataType::DT_INT16,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001\000")),
        },
        {
            "int32",
            tensorflow::test::AsScalar<int32_t>(1),
            testing::CreateArrayContent(federated_language::DataType::DT_INT32,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001\000\000\000")),
        },
        {
            "int64",
            tensorflow::test::AsScalar<int64_t>(1),
            testing::CreateArrayContent(
                federated_language::DataType::DT_INT64,
                testing::CreateArrayShape({}),
                CONTENT("\001\000\000\000\000\000\000\000")),
        },
        {
            "uint8",
            tensorflow::test::AsScalar<uint8_t>(1),
            testing::CreateArrayContent(federated_language::DataType::DT_UINT8,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001")),
        },
        {
            "uint16",
            tensorflow::test::AsScalar<uint16_t>(1),
            testing::CreateArrayContent(federated_language::DataType::DT_UINT16,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001\000")),
        },
        {
            "uint32",
            tensorflow::test::AsScalar<uint32_t>(1),
            testing::CreateArrayContent(federated_language::DataType::DT_UINT32,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001\000\000\000")),
        },
        {
            "uint64",
            tensorflow::test::AsScalar<uint64_t>(1),
            testing::CreateArrayContent(
                federated_language::DataType::DT_UINT64,
                testing::CreateArrayShape({}),
                CONTENT("\001\000\000\000\000\000\000\000")),
        },
        {
            "float16",
            tensorflow::test::AsScalar(Eigen::half{1.0}),
            testing::CreateArrayContent(federated_language::DataType::DT_HALF,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\000<")),
        },
        {
            "float32",
            tensorflow::test::AsScalar<float>(1.0),
            testing::CreateArrayContent(federated_language::DataType::DT_FLOAT,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\000\000\200?")),
        },
        {
            "float64",
            tensorflow::test::AsScalar<double>(1.0),
            testing::CreateArrayContent(
                federated_language::DataType::DT_DOUBLE,
                testing::CreateArrayShape({}),
                CONTENT("\000\000\000\000\000\000\360?")),
        },
        {
            "complex64",
            tensorflow::test::AsScalar(tensorflow::complex64{1.0, 1.0}),
            testing::CreateArrayContent(
                federated_language::DataType::DT_COMPLEX64,
                testing::CreateArrayShape({}),
                CONTENT("\000\000\200?\000\000\200?")),
        },
        {
            "complex128",
            tensorflow::test::AsScalar(tensorflow::complex128{1.0, 1.0}),
            testing::CreateArrayContent(
                federated_language::DataType::DT_COMPLEX128,
                testing::CreateArrayShape({}),
                CONTENT("\000\000\000\000\000\000\360?"
                        "\000\000\000\000\000\000\360?")),
        },
        {
            "bfloat16",
            tensorflow::test::AsScalar(Eigen::bfloat16{1.0}),
            testing::CreateArrayContent(
                federated_language::DataType::DT_BFLOAT16,
                testing::CreateArrayShape({}), CONTENT("\200?")),
        },
        {
            "array",
            tensorflow::test::AsTensor<int32_t>(
                {1, 2, 3, 4, 5, 6}, tensorflow::TensorShape({2, 3})),
            testing::CreateArrayContent(
                federated_language::DataType::DT_INT32,
                testing::CreateArrayShape({2, 3}),
                CONTENT("\001\000\000\000\002\000\000\000\003\000\000\000\004"
                        "\000\000\000\005\000\000\000\006\000\000\000")),
        },
    }),
    [](const ::testing::TestParamInfo<ArrayContentFromTensorTest::ParamType>&
           info) { return info.param.test_name; });

struct TensorFromArrayContentTestCase {
  std::string test_name;
  const federated_language::Array array_pb;
  const tensorflow::Tensor expected_tensor;
};

using TensorFromArrayContentTest =
    ::testing::TestWithParam<TensorFromArrayContentTestCase>;

TEST_P(TensorFromArrayContentTest, TestReturnsTensor) {
  const TensorFromArrayContentTestCase& test_case = GetParam();

  const tensorflow::Tensor& actual_tensor =
      TFF_ASSERT_OK(TensorFromArrayContent(test_case.array_pb));

  tensorflow::test::ExpectEqual(actual_tensor, test_case.expected_tensor);
}

#define CONTENT(s) absl::string_view(s, sizeof(s) - 1)

TEST(ArrayContentFromTensorTest, TestFailsOnStringTensor) {
  const tensorflow::Tensor tensor =
      tensorflow::test::AsTensor<tensorflow::tstring>({"a", "b"});

  const absl::StatusOr<federated_language::Array>& result =
      ArrayContentFromTensor(tensor);

  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(TensorFromArrayContentTest, TestFailsOnSizeMismatch) {
  const absl::StatusOr<federated_language::Array> array_pb =
      testing::CreateArrayContent(federated_language::DataType::DT_INT32,
                                  testing::CreateArrayShape({2, 3}),
                                  CONTENT("\001\000\000\000"));
  TFF_ASSERT_OK(array_pb);

  const absl::StatusOr<tensorflow::Tensor>& result =
      TensorFromArrayContent(*array_pb);

  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(TensorFromArrayContentTest, TestFailsOnInvalidVarintStringContent) {
  const absl::StatusOr<federated_language::Array> array_pb =
      testing::CreateArrayContent(federated_language::DataType::DT_STRING,
                                  testing::CreateArrayShape({2}),
                                  CONTENT("ab"));
  TFF_ASSERT_OK(array_pb);

  const absl::StatusOr<tensorflow::Tensor>& result =
      TensorFromArrayContent(*array_pb);

  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(TensorFromArrayContentTest, TestParsesValidVarintStringContent) {
  const tensorflow::Tensor expected_tensor =
      tensorflow::test::AsTensor<tensorflow::tstring>({"hello", "world"});
  tensorflow::TensorProto tensor_pb;
  expected_tensor.AsProtoTensorContent(&tensor_pb);

  federated_language::Array array_pb;
  array_pb.set_dtype(federated_language::DataType::DT_STRING);
  *array_pb.mutable_shape() = testing::CreateArrayShape({2});
  array_pb.set_content(tensor_pb.tensor_content());

  const tensorflow::Tensor actual_tensor =
      TFF_ASSERT_OK(TensorFromArrayContent(array_pb));

  tensorflow::test::ExpectEqual(actual_tensor, expected_tensor);
}

TEST(TensorFromArrayContentTest, TestFallsBackToStringList) {
  federated_language::Array array_pb = TFF_ASSERT_OK(
      testing::CreateArray(federated_language::DataType::DT_STRING,
                           testing::CreateArrayShape({2}), {"a", "b"}));
  array_pb.set_content("ab");

  const tensorflow::Tensor& actual_tensor =
      TFF_ASSERT_OK(TensorFromArrayContent(array_pb));

  const tensorflow::Tensor expected_tensor =
      tensorflow::test::AsTensor<tensorflow::tstring>({"a", "b"});
  tensorflow::test::ExpectEqual(actual_tensor, expected_tensor);
}

TEST(TensorFromArrayContentTest, TestFailsOnInvalidBooleanByte) {
  const absl::StatusOr<federated_language::Array> array_pb =
      testing::CreateArrayContent(federated_language::DataType::DT_BOOL,
                                  testing::CreateArrayShape({1}),
                                  CONTENT("\002"));
  TFF_ASSERT_OK(array_pb);

  const absl::StatusOr<tensorflow::Tensor>& result =
      TensorFromArrayContent(*array_pb);

  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(TensorFromArrayContentTest, TestFailsOnInvalidBooleanByteAtNonZeroIndex) {
  const absl::StatusOr<federated_language::Array> array_pb =
      testing::CreateArrayContent(federated_language::DataType::DT_BOOL,
                                  testing::CreateArrayShape({3}),
                                  CONTENT("\001\000\002"));
  TFF_ASSERT_OK(array_pb);

  const absl::StatusOr<tensorflow::Tensor>& result =
      TensorFromArrayContent(*array_pb);

  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(TensorFromArrayContentTest, TestReturnsEmptyTensor) {
  const absl::StatusOr<federated_language::Array> array_pb =
      testing::CreateArrayContent(federated_language::DataType::DT_INT32,
                                  testing::CreateArrayShape({0}), CONTENT(""));
  TFF_ASSERT_OK(array_pb);

  const absl::StatusOr<tensorflow::Tensor>& result =
      TensorFromArrayContent(*array_pb);

  TFF_ASSERT_OK(result);
  EXPECT_EQ(result->NumElements(), 0);
}

TEST(TensorFromArrayContentTest, TestReturnsDefaultStringTensorOnEmptyContent) {
  const absl::StatusOr<federated_language::Array> array_pb =
      testing::CreateArrayContent(federated_language::DataType::DT_STRING,
                                  testing::CreateArrayShape({10}), CONTENT(""));
  TFF_ASSERT_OK(array_pb);

  const absl::StatusOr<tensorflow::Tensor>& result =
      TensorFromArrayContent(*array_pb);

  TFF_ASSERT_OK(result);
  EXPECT_EQ(result->dtype(), tensorflow::DT_STRING);
  EXPECT_EQ(result->NumElements(), 10);
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(result->flat<tensorflow::tstring>()(i), "");
  }
}

TEST(TensorFromArrayContentTest, TestSucceedsWithFragmentedCord) {
  const absl::Cord content = absl::MakeFragmentedCord({
      absl::string_view("\001\000\000\000\002\000\000\000", 8),
      absl::string_view("\003\000\000\000\004\000\000\000", 8),
      absl::string_view("\005\000\000\000\006\000\000\000", 8),
  });
  int num_chunks = 0;
  for (absl::string_view chunk : content.Chunks()) {
    (void)chunk;
    ++num_chunks;
  }
  ASSERT_GT(num_chunks, 1);

  const absl::StatusOr<federated_language::Array> array_pb =
      testing::CreateArrayContent(federated_language::DataType::DT_INT32,
                                  testing::CreateArrayShape({2, 3}), content);
  TFF_ASSERT_OK(array_pb);

  const absl::StatusOr<tensorflow::Tensor>& actual_tensor =
      TensorFromArrayContent(*array_pb);
  TFF_ASSERT_OK(actual_tensor);

  const tensorflow::Tensor expected_tensor =
      tensorflow::test::AsTensor<int32_t>({1, 2, 3, 4, 5, 6},
                                          tensorflow::TensorShape({2, 3}));
  tensorflow::test::ExpectEqual(*actual_tensor, expected_tensor);
}

INSTANTIATE_TEST_SUITE_P(
    TensorFromArrayContentTestSuiteInstantiation, TensorFromArrayContentTest,
    ::testing::ValuesIn<TensorFromArrayContentTestCase>({
        {
            "bool",
            testing::CreateArrayContent(federated_language::DataType::DT_BOOL,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001")),
            tensorflow::test::AsScalar(true),
        },
        {
            "bool_array",
            testing::CreateArrayContent(federated_language::DataType::DT_BOOL,
                                        testing::CreateArrayShape({2, 2}),
                                        CONTENT("\001\000\000\001")),
            tensorflow::test::AsTensor<bool>({true, false, false, true},
                                             tensorflow::TensorShape({2, 2})),
        },
        {
            "int8",
            testing::CreateArrayContent(federated_language::DataType::DT_INT8,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001")),
            tensorflow::test::AsScalar<int8_t>(1),
        },
        {
            "int16",
            testing::CreateArrayContent(federated_language::DataType::DT_INT16,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001\000")),
            tensorflow::test::AsScalar<int16_t>(1),
        },
        {
            "int32",
            testing::CreateArrayContent(federated_language::DataType::DT_INT32,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001\000\000\000")),
            tensorflow::test::AsScalar<int32_t>(1),
        },
        {
            "int64",
            testing::CreateArrayContent(
                federated_language::DataType::DT_INT64,
                testing::CreateArrayShape({}),
                CONTENT("\001\000\000\000\000\000\000\000")),
            tensorflow::test::AsScalar<int64_t>(1),
        },
        {
            "uint8",
            testing::CreateArrayContent(federated_language::DataType::DT_UINT8,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001")),
            tensorflow::test::AsScalar<uint8_t>(1),
        },
        {
            "uint16",
            testing::CreateArrayContent(federated_language::DataType::DT_UINT16,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001\000")),
            tensorflow::test::AsScalar<uint16_t>(1),
        },
        {
            "uint32",
            testing::CreateArrayContent(federated_language::DataType::DT_UINT32,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001\000\000\000")),
            tensorflow::test::AsScalar<uint32_t>(1),
        },
        {
            "uint64",
            testing::CreateArrayContent(
                federated_language::DataType::DT_UINT64,
                testing::CreateArrayShape({}),
                CONTENT("\001\000\000\000\000\000\000\000")),
            tensorflow::test::AsScalar<uint64_t>(1),
        },
        {
            "float16",
            testing::CreateArrayContent(federated_language::DataType::DT_HALF,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\000<")),
            tensorflow::test::AsScalar(Eigen::half{1.0}),
        },
        {
            "float32",
            testing::CreateArrayContent(federated_language::DataType::DT_FLOAT,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\000\000\200?")),
            tensorflow::test::AsScalar<float>(1.0),
        },
        {
            "float64",
            testing::CreateArrayContent(
                federated_language::DataType::DT_DOUBLE,
                testing::CreateArrayShape({}),
                CONTENT("\000\000\000\000\000\000\360?")),
            tensorflow::test::AsScalar<double>(1.0),
        },
        {
            "complex64",
            testing::CreateArrayContent(
                federated_language::DataType::DT_COMPLEX64,
                testing::CreateArrayShape({}),
                CONTENT("\000\000\200?\000\000\200?")),
            tensorflow::test::AsScalar(tensorflow::complex64{1.0, 1.0}),
        },
        {
            "complex128",
            testing::CreateArrayContent(
                federated_language::DataType::DT_COMPLEX128,
                testing::CreateArrayShape({}),
                CONTENT("\000\000\000\000\000\000\360?"
                        "\000\000\000\000\000\000\360?")),
            tensorflow::test::AsScalar(tensorflow::complex128{1.0, 1.0}),
        },
        {
            "bfloat16",
            testing::CreateArrayContent(
                federated_language::DataType::DT_BFLOAT16,
                testing::CreateArrayShape({}), CONTENT("\200?")),
            tensorflow::test::AsScalar(Eigen::bfloat16{1.0}),
        },
        {
            "array",
            testing::CreateArrayContent(
                federated_language::DataType::DT_INT32,
                testing::CreateArrayShape({2, 3}),
                CONTENT("\001\000\000\000\002\000\000\000\003\000\000\000\004"
                        "\000\000\000\005\000\000\000\006\000\000\000")),
            tensorflow::test::AsTensor<int32_t>(
                {1, 2, 3, 4, 5, 6}, tensorflow::TensorShape({2, 3})),
        },
    }),
    [](const ::testing::TestParamInfo<TensorFromArrayContentTest::ParamType>&
           info) { return info.param.test_name; });

}  // namespace
}  // namespace tensorflow_federated

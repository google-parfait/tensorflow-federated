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

#include <algorithm>
#include <complex>
#include <cstdint>
#include <string>
#include <type_traits>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "federated_language/proto/array.pb.h"
#include "federated_language/proto/computation.pb.h"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow_federated/cc/core/impl/executors/array_shape_test_utils.h"
#include "tensorflow_federated/cc/core/impl/executors/array_test_utils.h"
#include "tensorflow_federated/cc/testing/protobuf_matchers.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

namespace tensorflow_federated {
namespace {

template <typename T>
tensorflow::Tensor AsScalar(const T& val) {
  tensorflow::Tensor ret(tensorflow::DataTypeToEnum<T>::value, {});
  ret.scalar<T>()() = val;
  return ret;
}

template <typename T>
tensorflow::Tensor AsTensor(absl::Span<const T> vals,
                            const tensorflow::TensorShape& shape) {
  tensorflow::Tensor ret(tensorflow::DataTypeToEnum<T>::value, shape);
  auto flat = ret.flat<T>();
  CHECK_EQ(flat.size(), vals.size());
  std::copy_n(vals.data(), vals.size(), flat.data());
  return ret;
}

template <typename TensorType>
void TensorToProtoContent(const TensorType& tensor,
                          tensorflow::TensorProto* proto) {
  if constexpr (std::is_same_v<decltype(tensor.AsProtoTensorContent(proto)),
                               bool>) {
    ASSERT_TRUE(tensor.AsProtoTensorContent(proto));
  } else {
    tensor.AsProtoTensorContent(proto);
  }
}

void ExpectTensorEqual(const tensorflow::Tensor& actual,
                       const tensorflow::Tensor& expected) {
  EXPECT_EQ(actual.dtype(), expected.dtype());
  EXPECT_EQ(actual.shape(), expected.shape());
  tensorflow::TensorProto actual_proto;
  ASSERT_NO_FATAL_FAILURE(TensorToProtoContent(actual, &actual_proto));
  tensorflow::TensorProto expected_proto;
  ASSERT_NO_FATAL_FAILURE(TensorToProtoContent(expected, &expected_proto));
  EXPECT_THAT(actual_proto, testing::EqualsProto(expected_proto));
}

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
            AsScalar(true),
            testing::CreateArray(federated_language::DataType::DT_BOOL,
                                 testing::CreateArrayShape({}), {true})
                .value(),
        },
        {
            "int8",
            AsScalar<int8_t>(1),
            testing::CreateArray(federated_language::DataType::DT_INT8,
                                 testing::CreateArrayShape({}), {1})
                .value(),
        },
        {
            "int16",
            AsScalar<int16_t>(1),
            testing::CreateArray(federated_language::DataType::DT_INT16,
                                 testing::CreateArrayShape({}), {1})
                .value(),
        },
        {
            "int32",
            AsScalar<int32_t>(1),
            testing::CreateArray(federated_language::DataType::DT_INT32,
                                 testing::CreateArrayShape({}), {1})
                .value(),
        },
        {
            "int64",
            AsScalar<int64_t>(1),
            testing::CreateArray(federated_language::DataType::DT_INT64,
                                 testing::CreateArrayShape({}), {1})
                .value(),
        },
        {
            "uint8",
            AsScalar<uint8_t>(1),
            testing::CreateArray(federated_language::DataType::DT_UINT8,
                                 testing::CreateArrayShape({}), {1})
                .value(),
        },
        {
            "uint16",
            AsScalar<uint16_t>(1),
            testing::CreateArray(federated_language::DataType::DT_UINT16,
                                 testing::CreateArrayShape({}), {1})
                .value(),
        },
        {
            "uint32",
            AsScalar<uint32_t>(1),
            testing::CreateArray(federated_language::DataType::DT_UINT32,
                                 testing::CreateArrayShape({}), {1})
                .value(),
        },
        {
            "uint64",
            AsScalar<uint64_t>(1),
            testing::CreateArray(federated_language::DataType::DT_UINT64,
                                 testing::CreateArrayShape({}), {1})
                .value(),
        },
        {
            "float16",
            AsScalar(Eigen::half{1.0}),
            testing::CreateArray(federated_language::DataType::DT_HALF,
                                 testing::CreateArrayShape({}),
                                 {Eigen::half{1.0}})
                .value(),
        },
        {
            "float32",
            AsScalar<float>(1.0),
            testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                 testing::CreateArrayShape({}), {1.0})
                .value(),
        },
        {
            "float64",
            AsScalar<double>(1.0),
            testing::CreateArray(federated_language::DataType::DT_DOUBLE,
                                 testing::CreateArrayShape({}), {1.0})
                .value(),
        },
        {
            "complex64",
            AsScalar(tensorflow::complex64{1.0, 1.0}),
            testing::CreateArray(federated_language::DataType::DT_COMPLEX64,
                                 testing::CreateArrayShape({}),
                                 {std::complex<float>(1.0, 1.0)})
                .value(),
        },
        {
            "complex128",
            AsScalar(tensorflow::complex128{1.0, 1.0}),
            testing::CreateArray(federated_language::DataType::DT_COMPLEX128,
                                 testing::CreateArrayShape({}),
                                 {std::complex<double>(1.0, 1.0)})
                .value(),
        },
        {
            "bfloat16",
            AsScalar(Eigen::bfloat16{1.0}),
            testing::CreateArray(federated_language::DataType::DT_BFLOAT16,
                                 testing::CreateArrayShape({}),
                                 {Eigen::bfloat16{1.0}})
                .value(),
        },
        {
            "string",
            AsScalar<tensorflow::tstring>("a"),
            testing::CreateArray(federated_language::DataType::DT_STRING,
                                 testing::CreateArrayShape({}), {"a"})
                .value(),
        },
        {
            "array",
            AsTensor<int32_t>({1, 2, 3, 4, 5, 6},
                              tensorflow::TensorShape({2, 3})),
            testing::CreateArray(federated_language::DataType::DT_INT32,
                                 testing::CreateArrayShape({2, 3}),
                                 {1, 2, 3, 4, 5, 6})
                .value(),
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

  ExpectTensorEqual(actual_tensor, test_case.expected_tensor);
}

INSTANTIATE_TEST_SUITE_P(
    TensorFromArrayTestSuiteInstantiation, TensorFromArrayTest,
    ::testing::ValuesIn<TensorFromArrayTestCase>({
        {
            "bool",
            testing::CreateArray(federated_language::DataType::DT_BOOL,
                                 testing::CreateArrayShape({}), {true})
                .value(),
            AsScalar(true),
        },
        {
            "int8",
            testing::CreateArray(federated_language::DataType::DT_INT8,
                                 testing::CreateArrayShape({}), {1})
                .value(),
            AsScalar<int8_t>(1),
        },
        {
            "int16",
            testing::CreateArray(federated_language::DataType::DT_INT16,
                                 testing::CreateArrayShape({}), {1})
                .value(),
            AsScalar<int16_t>(1),
        },
        {
            "int32",
            testing::CreateArray(federated_language::DataType::DT_INT32,
                                 testing::CreateArrayShape({}), {1})
                .value(),
            AsScalar<int32_t>(1),
        },
        {
            "int64",
            testing::CreateArray(federated_language::DataType::DT_INT64,
                                 testing::CreateArrayShape({}), {1})
                .value(),
            AsScalar<int64_t>(1),
        },
        {
            "uint8",
            testing::CreateArray(federated_language::DataType::DT_UINT8,
                                 testing::CreateArrayShape({}), {1})
                .value(),
            AsScalar<uint8_t>(1),
        },
        {
            "uint16",
            testing::CreateArray(federated_language::DataType::DT_UINT16,
                                 testing::CreateArrayShape({}), {1})
                .value(),
            AsScalar<uint16_t>(1),
        },
        {
            "uint32",
            testing::CreateArray(federated_language::DataType::DT_UINT32,
                                 testing::CreateArrayShape({}), {1})
                .value(),
            AsScalar<uint32_t>(1),
        },
        {
            "uint64",
            testing::CreateArray(federated_language::DataType::DT_UINT64,
                                 testing::CreateArrayShape({}), {1})
                .value(),
            AsScalar<uint64_t>(1),
        },
        {
            "float16",
            testing::CreateArray(federated_language::DataType::DT_HALF,
                                 testing::CreateArrayShape({}),
                                 {Eigen::half{1.0}})
                .value(),
            AsScalar(Eigen::half{1.0}),
        },
        {
            "float32",
            testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                 testing::CreateArrayShape({}), {1.0})
                .value(),
            AsScalar<float>(1.0),
        },
        {
            "float64",
            testing::CreateArray(federated_language::DataType::DT_DOUBLE,
                                 testing::CreateArrayShape({}), {1.0})
                .value(),
            AsScalar<double>(1.0),
        },
        {
            "complex64",
            testing::CreateArray(federated_language::DataType::DT_COMPLEX64,
                                 testing::CreateArrayShape({}),
                                 {std::complex<float>(1.0, 1.0)})
                .value(),
            AsScalar(tensorflow::complex64{1.0, 1.0}),
        },
        {
            "complex128",
            testing::CreateArray(federated_language::DataType::DT_COMPLEX128,
                                 testing::CreateArrayShape({}),
                                 {std::complex<double>(1.0, 1.0)})
                .value(),
            AsScalar(tensorflow::complex128{1.0, 1.0}),
        },
        {
            "bfloat16",
            testing::CreateArray(federated_language::DataType::DT_BFLOAT16,
                                 testing::CreateArrayShape({}),
                                 {Eigen::bfloat16{1.0}})
                .value(),
            AsScalar(Eigen::bfloat16{1.0}),
        },
        {
            "string",
            testing::CreateArray(federated_language::DataType::DT_STRING,
                                 testing::CreateArrayShape({}), {"a"})
                .value(),
            AsScalar<tensorflow::tstring>("a"),
        },
        {
            "array",
            testing::CreateArray(federated_language::DataType::DT_INT32,
                                 testing::CreateArrayShape({2, 3}),
                                 {1, 2, 3, 4, 5, 6})
                .value(),
            AsTensor<int32_t>({1, 2, 3, 4, 5, 6},
                              tensorflow::TensorShape({2, 3})),
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
            AsScalar(true),
            testing::CreateArrayContent(federated_language::DataType::DT_BOOL,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001"))
                .value(),
        },
        {
            "int8",
            AsScalar<int8_t>(1),
            testing::CreateArrayContent(federated_language::DataType::DT_INT8,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001"))
                .value(),
        },
        {
            "int16",
            AsScalar<int16_t>(1),
            testing::CreateArrayContent(federated_language::DataType::DT_INT16,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001\000"))
                .value(),
        },
        {
            "int32",
            AsScalar<int32_t>(1),
            testing::CreateArrayContent(federated_language::DataType::DT_INT32,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001\000\000\000"))
                .value(),
        },
        {
            "int64",
            AsScalar<int64_t>(1),
            testing::CreateArrayContent(
                federated_language::DataType::DT_INT64,
                testing::CreateArrayShape({}),
                CONTENT("\001\000\000\000\000\000\000\000"))
                .value(),
        },
        {
            "uint8",
            AsScalar<uint8_t>(1),
            testing::CreateArrayContent(federated_language::DataType::DT_UINT8,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001"))
                .value(),
        },
        {
            "uint16",
            AsScalar<uint16_t>(1),
            testing::CreateArrayContent(federated_language::DataType::DT_UINT16,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001\000"))
                .value(),
        },
        {
            "uint32",
            AsScalar<uint32_t>(1),
            testing::CreateArrayContent(federated_language::DataType::DT_UINT32,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001\000\000\000"))
                .value(),
        },
        {
            "uint64",
            AsScalar<uint64_t>(1),
            testing::CreateArrayContent(
                federated_language::DataType::DT_UINT64,
                testing::CreateArrayShape({}),
                CONTENT("\001\000\000\000\000\000\000\000"))
                .value(),
        },
        {
            "float16",
            AsScalar(Eigen::half{1.0}),
            testing::CreateArrayContent(federated_language::DataType::DT_HALF,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\000<"))
                .value(),
        },
        {
            "float32",
            AsScalar<float>(1.0),
            testing::CreateArrayContent(federated_language::DataType::DT_FLOAT,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\000\000\200?"))
                .value(),
        },
        {
            "float64",
            AsScalar<double>(1.0),
            testing::CreateArrayContent(
                federated_language::DataType::DT_DOUBLE,
                testing::CreateArrayShape({}),
                CONTENT("\000\000\000\000\000\000\360?"))
                .value(),
        },
        {
            "complex64",
            AsScalar(tensorflow::complex64{1.0, 1.0}),
            testing::CreateArrayContent(
                federated_language::DataType::DT_COMPLEX64,
                testing::CreateArrayShape({}),
                CONTENT("\000\000\200?\000\000\200?"))
                .value(),
        },
        {
            "complex128",
            AsScalar(tensorflow::complex128{1.0, 1.0}),
            testing::CreateArrayContent(
                federated_language::DataType::DT_COMPLEX128,
                testing::CreateArrayShape({}),
                CONTENT("\000\000\000\000\000\000\360?"
                        "\000\000\000\000\000\000\360?"))
                .value(),
        },
        {
            "bfloat16",
            AsScalar(Eigen::bfloat16{1.0}),
            testing::CreateArrayContent(
                federated_language::DataType::DT_BFLOAT16,
                testing::CreateArrayShape({}), CONTENT("\200?"))
                .value(),
        },
        {
            "array",
            AsTensor<int32_t>({1, 2, 3, 4, 5, 6},
                              tensorflow::TensorShape({2, 3})),
            testing::CreateArrayContent(
                federated_language::DataType::DT_INT32,
                testing::CreateArrayShape({2, 3}),
                CONTENT("\001\000\000\000\002\000\000\000\003\000\000\000\004"
                        "\000\000\000\005\000\000\000\006\000\000\000"))
                .value(),
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

  ExpectTensorEqual(actual_tensor, test_case.expected_tensor);
}

#define CONTENT(s) absl::string_view(s, sizeof(s) - 1)

INSTANTIATE_TEST_SUITE_P(
    TensorFromArrayContentTestSuiteInstantiation, TensorFromArrayContentTest,
    ::testing::ValuesIn<TensorFromArrayContentTestCase>({
        {
            "bool",
            testing::CreateArrayContent(federated_language::DataType::DT_BOOL,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001"))
                .value(),
            AsScalar(true),
        },
        {
            "int8",
            testing::CreateArrayContent(federated_language::DataType::DT_INT8,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001"))
                .value(),
            AsScalar<int8_t>(1),
        },
        {
            "int16",
            testing::CreateArrayContent(federated_language::DataType::DT_INT16,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001\000"))
                .value(),
            AsScalar<int16_t>(1),
        },
        {
            "int32",
            testing::CreateArrayContent(federated_language::DataType::DT_INT32,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001\000\000\000"))
                .value(),
            AsScalar<int32_t>(1),
        },
        {
            "int64",
            testing::CreateArrayContent(
                federated_language::DataType::DT_INT64,
                testing::CreateArrayShape({}),
                CONTENT("\001\000\000\000\000\000\000\000"))
                .value(),
            AsScalar<int64_t>(1),
        },
        {
            "uint8",
            testing::CreateArrayContent(federated_language::DataType::DT_UINT8,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001"))
                .value(),
            AsScalar<uint8_t>(1),
        },
        {
            "uint16",
            testing::CreateArrayContent(federated_language::DataType::DT_UINT16,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001\000"))
                .value(),
            AsScalar<uint16_t>(1),
        },
        {
            "uint32",
            testing::CreateArrayContent(federated_language::DataType::DT_UINT32,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\001\000\000\000"))
                .value(),
            AsScalar<uint32_t>(1),
        },
        {
            "uint64",
            testing::CreateArrayContent(
                federated_language::DataType::DT_UINT64,
                testing::CreateArrayShape({}),
                CONTENT("\001\000\000\000\000\000\000\000"))
                .value(),
            AsScalar<uint64_t>(1),
        },
        {
            "float16",
            testing::CreateArrayContent(federated_language::DataType::DT_HALF,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\000<"))
                .value(),
            AsScalar(Eigen::half{1.0}),
        },
        {
            "float32",
            testing::CreateArrayContent(federated_language::DataType::DT_FLOAT,
                                        testing::CreateArrayShape({}),
                                        CONTENT("\000\000\200?"))
                .value(),
            AsScalar<float>(1.0),
        },
        {
            "float64",
            testing::CreateArrayContent(
                federated_language::DataType::DT_DOUBLE,
                testing::CreateArrayShape({}),
                CONTENT("\000\000\000\000\000\000\360?"))
                .value(),
            AsScalar<double>(1.0),
        },
        {
            "complex64",
            testing::CreateArrayContent(
                federated_language::DataType::DT_COMPLEX64,
                testing::CreateArrayShape({}),
                CONTENT("\000\000\200?\000\000\200?"))
                .value(),
            AsScalar(tensorflow::complex64{1.0, 1.0}),
        },
        {
            "complex128",
            testing::CreateArrayContent(
                federated_language::DataType::DT_COMPLEX128,
                testing::CreateArrayShape({}),
                CONTENT("\000\000\000\000\000\000\360?"
                        "\000\000\000\000\000\000\360?"))
                .value(),
            AsScalar(tensorflow::complex128{1.0, 1.0}),
        },
        {
            "bfloat16",
            testing::CreateArrayContent(
                federated_language::DataType::DT_BFLOAT16,
                testing::CreateArrayShape({}), CONTENT("\200?"))
                .value(),
            AsScalar(Eigen::bfloat16{1.0}),
        },
        {
            "array",
            testing::CreateArrayContent(
                federated_language::DataType::DT_INT32,
                testing::CreateArrayShape({2, 3}),
                CONTENT("\001\000\000\000\002\000\000\000\003\000\000\000\004"
                        "\000\000\000\005\000\000\000\006\000\000\000"))
                .value(),
            AsTensor<int32_t>({1, 2, 3, 4, 5, 6},
                              tensorflow::TensorShape({2, 3})),
        },
    }),
    [](const ::testing::TestParamInfo<TensorFromArrayContentTest::ParamType>&
           info) { return info.param.test_name; });

}  // namespace
}  // namespace tensorflow_federated

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
#include <cstdlib>
#include <string>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
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
T ValueOrDie(absl::StatusOr<T> status_or) {
  CHECK(status_or.ok()) << status_or.status();
  return *std::move(status_or);
}
template <typename T>
tensorflow::Tensor AsScalar(T value) {
  tensorflow::Tensor t(tensorflow::DataTypeToEnum<T>::value,
                       tensorflow::TensorShape({}));
  t.scalar<T>()() = value;
  return t;
}

// Specialization for complex64
template <>
tensorflow::Tensor AsScalar<tensorflow::complex64>(
    tensorflow::complex64 value) {
  tensorflow::Tensor t(tensorflow::DT_COMPLEX64, tensorflow::TensorShape({}));
  t.scalar<tensorflow::complex64>()() = value;
  return t;
}

// Specialization for complex128
template <>
tensorflow::Tensor AsScalar<tensorflow::complex128>(
    tensorflow::complex128 value) {
  tensorflow::Tensor t(tensorflow::DT_COMPLEX128, tensorflow::TensorShape({}));
  t.scalar<tensorflow::complex128>()() = value;
  return t;
}

template <typename T>
tensorflow::Tensor AsTensor(std::initializer_list<T> values,
                            const tensorflow::TensorShape& shape) {
  tensorflow::Tensor t(tensorflow::DataTypeToEnum<T>::value, shape);
  auto flat = t.flat<T>();
  if (flat.size() != values.size()) {
    std::abort();
  }
  int i = 0;
  for (const auto& v : values) {
    flat(i++) = v;
  }
  return t;
}

void ExpectEqual(const tensorflow::Tensor& t1, const tensorflow::Tensor& t2) {
  EXPECT_EQ(t1.dtype(), t2.dtype());
  EXPECT_EQ(t1.shape(), t2.shape());
  if (t1.dtype() == tensorflow::DT_STRING) {
    ASSERT_EQ(t1.NumElements(), t2.NumElements());
    for (int i = 0; i < t1.NumElements(); ++i) {
      EXPECT_EQ(t1.flat<tensorflow::tstring>()(i),
                t2.flat<tensorflow::tstring>()(i));
    }
  } else {
    EXPECT_EQ(t1.tensor_data(), t2.tensor_data());
  }
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
            ValueOrDie(
                testing::CreateArray(federated_language::DataType::DT_BOOL,
                                     testing::CreateArrayShape({}), {true})),
        },
        {
            "int8",
            AsScalar<int8_t>(1),
            ValueOrDie(
                testing::CreateArray(federated_language::DataType::DT_INT8,
                                     testing::CreateArrayShape({}), {1})),
        },
        {
            "int16",
            AsScalar<int16_t>(1),
            ValueOrDie(
                testing::CreateArray(federated_language::DataType::DT_INT16,
                                     testing::CreateArrayShape({}), {1})),
        },
        {
            "int32",
            AsScalar<int32_t>(1),
            ValueOrDie(
                testing::CreateArray(federated_language::DataType::DT_INT32,
                                     testing::CreateArrayShape({}), {1})),
        },
        {
            "int64",
            AsScalar<int64_t>(1),
            ValueOrDie(
                testing::CreateArray(federated_language::DataType::DT_INT64,
                                     testing::CreateArrayShape({}), {1})),
        },
        {
            "uint8",
            AsScalar<uint8_t>(1),
            ValueOrDie(
                testing::CreateArray(federated_language::DataType::DT_UINT8,
                                     testing::CreateArrayShape({}), {1})),
        },
        {
            "uint16",
            AsScalar<uint16_t>(1),
            ValueOrDie(
                testing::CreateArray(federated_language::DataType::DT_UINT16,
                                     testing::CreateArrayShape({}), {1})),
        },
        {
            "uint32",
            AsScalar<uint32_t>(1),
            ValueOrDie(
                testing::CreateArray(federated_language::DataType::DT_UINT32,
                                     testing::CreateArrayShape({}), {1})),
        },
        {
            "uint64",
            AsScalar<uint64_t>(1),
            ValueOrDie(
                testing::CreateArray(federated_language::DataType::DT_UINT64,
                                     testing::CreateArrayShape({}), {1})),
        },
        {
            "float16",
            AsScalar(Eigen::half{1.0}),
            ValueOrDie(testing::CreateArray(
                federated_language::DataType::DT_HALF,
                testing::CreateArrayShape({}), {Eigen::half{1.0}})),
        },
        {
            "float32",
            AsScalar<float>(1.0),
            ValueOrDie(
                testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                     testing::CreateArrayShape({}), {1.0})),
        },
        {
            "float64",
            AsScalar<double>(1.0),
            ValueOrDie(
                testing::CreateArray(federated_language::DataType::DT_DOUBLE,
                                     testing::CreateArrayShape({}), {1.0})),
        },
        {
            "complex64",
            AsScalar(tensorflow::complex64{1.0, 1.0}),
            ValueOrDie(
                testing::CreateArray(federated_language::DataType::DT_COMPLEX64,
                                     testing::CreateArrayShape({}),
                                     {std::complex<float>(1.0, 1.0)})),
        },
        {
            "complex128",
            AsScalar(tensorflow::complex128{1.0, 1.0}),
            ValueOrDie(testing::CreateArray(
                federated_language::DataType::DT_COMPLEX128,
                testing::CreateArrayShape({}),
                {std::complex<double>(1.0, 1.0)})),
        },
        {
            "bfloat16",
            AsScalar(Eigen::bfloat16{1.0}),
            ValueOrDie(testing::CreateArray(
                federated_language::DataType::DT_BFLOAT16,
                testing::CreateArrayShape({}), {Eigen::bfloat16{1.0}})),
        },
        {
            "string",
            AsScalar<tensorflow::tstring>("a"),
            ValueOrDie(
                testing::CreateArray(federated_language::DataType::DT_STRING,
                                     testing::CreateArrayShape({}), {"a"})),
        },
        {
            "array",
            AsTensor<int32_t>({1, 2, 3, 4, 5, 6},
                              tensorflow::TensorShape({2, 3})),
            ValueOrDie(testing::CreateArray(
                federated_language::DataType::DT_INT32,
                testing::CreateArrayShape({2, 3}), {1, 2, 3, 4, 5, 6})),
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

  ExpectEqual(actual_tensor, test_case.expected_tensor);
}

INSTANTIATE_TEST_SUITE_P(
    TensorFromArrayTestSuiteInstantiation, TensorFromArrayTest,
    ::testing::ValuesIn<TensorFromArrayTestCase>({
        {
            "bool",
            ValueOrDie(
                testing::CreateArray(federated_language::DataType::DT_BOOL,
                                     testing::CreateArrayShape({}), {true})),
            AsScalar(true),
        },
        {
            "int8",
            ValueOrDie(
                testing::CreateArray(federated_language::DataType::DT_INT8,
                                     testing::CreateArrayShape({}), {1})),
            AsScalar<int8_t>(1),
        },
        {
            "int16",
            ValueOrDie(
                testing::CreateArray(federated_language::DataType::DT_INT16,
                                     testing::CreateArrayShape({}), {1})),
            AsScalar<int16_t>(1),
        },
        {
            "int32",
            ValueOrDie(
                testing::CreateArray(federated_language::DataType::DT_INT32,
                                     testing::CreateArrayShape({}), {1})),
            AsScalar<int32_t>(1),
        },
        {
            "int64",
            ValueOrDie(
                testing::CreateArray(federated_language::DataType::DT_INT64,
                                     testing::CreateArrayShape({}), {1})),
            AsScalar<int64_t>(1),
        },
        {
            "uint8",
            ValueOrDie(
                testing::CreateArray(federated_language::DataType::DT_UINT8,
                                     testing::CreateArrayShape({}), {1})),
            AsScalar<uint8_t>(1),
        },
        {
            "uint16",
            ValueOrDie(
                testing::CreateArray(federated_language::DataType::DT_UINT16,
                                     testing::CreateArrayShape({}), {1})),
            AsScalar<uint16_t>(1),
        },
        {
            "uint32",
            ValueOrDie(
                testing::CreateArray(federated_language::DataType::DT_UINT32,
                                     testing::CreateArrayShape({}), {1})),
            AsScalar<uint32_t>(1),
        },
        {
            "uint64",
            ValueOrDie(
                testing::CreateArray(federated_language::DataType::DT_UINT64,
                                     testing::CreateArrayShape({}), {1})),
            AsScalar<uint64_t>(1),
        },
        {
            "float16",
            ValueOrDie(testing::CreateArray(
                federated_language::DataType::DT_HALF,
                testing::CreateArrayShape({}), {Eigen::half{1.0}})),
            AsScalar(Eigen::half{1.0}),
        },
        {
            "float32",
            ValueOrDie(
                testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                     testing::CreateArrayShape({}), {1.0})),
            AsScalar<float>(1.0),
        },
        {
            "float64",
            ValueOrDie(
                testing::CreateArray(federated_language::DataType::DT_DOUBLE,
                                     testing::CreateArrayShape({}), {1.0})),
            AsScalar<double>(1.0),
        },
        {
            "complex64",
            ValueOrDie(
                testing::CreateArray(federated_language::DataType::DT_COMPLEX64,
                                     testing::CreateArrayShape({}),
                                     {std::complex<float>(1.0, 1.0)})),
            AsScalar(tensorflow::complex64{1.0, 1.0}),
        },
        {
            "complex128",
            ValueOrDie(testing::CreateArray(
                federated_language::DataType::DT_COMPLEX128,
                testing::CreateArrayShape({}),
                {std::complex<double>(1.0, 1.0)})),
            AsScalar(tensorflow::complex128{1.0, 1.0}),
        },
        {
            "bfloat16",
            ValueOrDie(testing::CreateArray(
                federated_language::DataType::DT_BFLOAT16,
                testing::CreateArrayShape({}), {Eigen::bfloat16{1.0}})),
            AsScalar(Eigen::bfloat16{1.0}),
        },
        {
            "string",
            ValueOrDie(
                testing::CreateArray(federated_language::DataType::DT_STRING,
                                     testing::CreateArrayShape({}), {"a"})),
            AsScalar<tensorflow::tstring>("a"),
        },
        {
            "array",
            ValueOrDie(testing::CreateArray(
                federated_language::DataType::DT_INT32,
                testing::CreateArrayShape({2, 3}), {1, 2, 3, 4, 5, 6})),
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
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_BOOL,
                testing::CreateArrayShape({}), CONTENT("\001"))),
        },
        {
            "int8",
            AsScalar<int8_t>(1),
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_INT8,
                testing::CreateArrayShape({}), CONTENT("\001"))),
        },
        {
            "int16",
            AsScalar<int16_t>(1),
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_INT16,
                testing::CreateArrayShape({}), CONTENT("\001\000"))),
        },
        {
            "int32",
            AsScalar<int32_t>(1),
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_INT32,
                testing::CreateArrayShape({}), CONTENT("\001\000\000\000"))),
        },
        {
            "int64",
            AsScalar<int64_t>(1),
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_INT64,
                testing::CreateArrayShape({}),
                CONTENT("\001\000\000\000\000\000\000\000"))),
        },
        {
            "uint8",
            AsScalar<uint8_t>(1),
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_UINT8,
                testing::CreateArrayShape({}), CONTENT("\001"))),
        },
        {
            "uint16",
            AsScalar<uint16_t>(1),
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_UINT16,
                testing::CreateArrayShape({}), CONTENT("\001\000"))),
        },
        {
            "uint32",
            AsScalar<uint32_t>(1),
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_UINT32,
                testing::CreateArrayShape({}), CONTENT("\001\000\000\000"))),
        },
        {
            "uint64",
            AsScalar<uint64_t>(1),
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_UINT64,
                testing::CreateArrayShape({}),
                CONTENT("\001\000\000\000\000\000\000\000"))),
        },
        {
            "float16",
            AsScalar(Eigen::half{1.0}),
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_HALF,
                testing::CreateArrayShape({}), CONTENT("\000<"))),
        },
        {
            "float32",
            AsScalar<float>(1.0),
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_FLOAT,
                testing::CreateArrayShape({}), CONTENT("\000\000\200?"))),
        },
        {
            "float64",
            AsScalar<double>(1.0),
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_DOUBLE,
                testing::CreateArrayShape({}),
                CONTENT("\000\000\000\000\000\000\360?"))),
        },
        {
            "complex64",
            AsScalar(tensorflow::complex64{1.0, 1.0}),
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_COMPLEX64,
                testing::CreateArrayShape({}),
                CONTENT("\000\000\200?\000\000\200?"))),
        },
        {
            "complex128",
            AsScalar(tensorflow::complex128{1.0, 1.0}),
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_COMPLEX128,
                testing::CreateArrayShape({}),
                CONTENT("\000\000\000\000\000\000\360?"
                        "\000\000\000\000\000\000\360?"))),
        },
        {
            "bfloat16",
            AsScalar(Eigen::bfloat16{1.0}),
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_BFLOAT16,
                testing::CreateArrayShape({}), CONTENT("\200?"))),
        },
        {
            "array",
            AsTensor<int32_t>({1, 2, 3, 4, 5, 6},
                              tensorflow::TensorShape({2, 3})),
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_INT32,
                testing::CreateArrayShape({2, 3}),
                CONTENT("\001\000\000\000\002\000\000\000\003\000\000\000\004"
                        "\000\000\000\005\000\000\000\006\000\000\000"))),
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

  ExpectEqual(actual_tensor, test_case.expected_tensor);
}

#define CONTENT(s) absl::string_view(s, sizeof(s) - 1)

INSTANTIATE_TEST_SUITE_P(
    TensorFromArrayContentTestSuiteInstantiation, TensorFromArrayContentTest,
    ::testing::ValuesIn<TensorFromArrayContentTestCase>({
        {
            "bool",
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_BOOL,
                testing::CreateArrayShape({}), CONTENT("\001"))),
            AsScalar(true),
        },
        {
            "int8",
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_INT8,
                testing::CreateArrayShape({}), CONTENT("\001"))),
            AsScalar<int8_t>(1),
        },
        {
            "int16",
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_INT16,
                testing::CreateArrayShape({}), CONTENT("\001\000"))),
            AsScalar<int16_t>(1),
        },
        {
            "int32",
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_INT32,
                testing::CreateArrayShape({}), CONTENT("\001\000\000\000"))),
            AsScalar<int32_t>(1),
        },
        {
            "int64",
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_INT64,
                testing::CreateArrayShape({}),
                CONTENT("\001\000\000\000\000\000\000\000"))),
            AsScalar<int64_t>(1),
        },
        {
            "uint8",
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_UINT8,
                testing::CreateArrayShape({}), CONTENT("\001"))),
            AsScalar<uint8_t>(1),
        },
        {
            "uint16",
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_UINT16,
                testing::CreateArrayShape({}), CONTENT("\001\000"))),
            AsScalar<uint16_t>(1),
        },
        {
            "uint32",
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_UINT32,
                testing::CreateArrayShape({}), CONTENT("\001\000\000\000"))),
            AsScalar<uint32_t>(1),
        },
        {
            "uint64",
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_UINT64,
                testing::CreateArrayShape({}),
                CONTENT("\001\000\000\000\000\000\000\000"))),
            AsScalar<uint64_t>(1),
        },
        {
            "float16",
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_HALF,
                testing::CreateArrayShape({}), CONTENT("\000<"))),
            AsScalar(Eigen::half{1.0}),
        },
        {
            "float32",
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_FLOAT,
                testing::CreateArrayShape({}), CONTENT("\000\000\200?"))),
            AsScalar<float>(1.0),
        },
        {
            "float64",
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_DOUBLE,
                testing::CreateArrayShape({}),
                CONTENT("\000\000\000\000\000\000\360?"))),
            AsScalar<double>(1.0),
        },
        {
            "complex64",
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_COMPLEX64,
                testing::CreateArrayShape({}),
                CONTENT("\000\000\200?\000\000\200?"))),
            AsScalar(tensorflow::complex64{1.0, 1.0}),
        },
        {
            "complex128",
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_COMPLEX128,
                testing::CreateArrayShape({}),
                CONTENT("\000\000\000\000\000\000\360?"
                        "\000\000\000\000\000\000\360?"))),
            AsScalar(tensorflow::complex128{1.0, 1.0}),
        },
        {
            "bfloat16",
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_BFLOAT16,
                testing::CreateArrayShape({}), CONTENT("\200?"))),
            AsScalar(Eigen::bfloat16{1.0}),
        },
        {
            "array",
            ValueOrDie(testing::CreateArrayContent(
                federated_language::DataType::DT_INT32,
                testing::CreateArrayShape({2, 3}),
                CONTENT("\001\000\000\000\002\000\000\000\003\000\000\000\004"
                        "\000\000\000\005\000\000\000\006\000\000\000"))),
            AsTensor<int32_t>({1, 2, 3, 4, 5, 6},
                              tensorflow::TensorShape({2, 3})),
        },
    }),
    [](const ::testing::TestParamInfo<TensorFromArrayContentTest::ParamType>&
           info) { return info.param.test_name; });

}  // namespace
}  // namespace tensorflow_federated

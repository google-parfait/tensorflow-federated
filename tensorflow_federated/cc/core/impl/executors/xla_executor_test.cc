/* Copyright 2022, The TensorFlow Federated Authors.

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

#include "tensorflow_federated/cc/core/impl/executors/xla_executor.h"

#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/status/status.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"
#include "tensorflow_federated/cc/core/impl/executors/value_test_utils.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"

namespace tensorflow_federated {
namespace {

using ::tensorflow_federated::testing::EqualsProto;
using ::tensorflow_federated::testing::StructV;
using ::tensorflow_federated::testing::TensorV;
using ::testing::HasSubstr;

inline absl::StatusOr<std::tuple<v0::Xla::Binding, int>> BindingFromType(
    v0::Type type, int next_unused_index) {
  switch (type.type_case()) {
    case v0::Type::kTensor: {
      v0::Xla::Binding binding;
      binding.mutable_tensor()->set_index(next_unused_index);
      return std::make_tuple(binding, next_unused_index + 1);
    }
    case v0::Type::kStruct: {
      v0::Xla::Binding binding;
      for (const auto& type_element : type.struct_().element()) {
        auto partial_binding =
            TFF_TRY(BindingFromType(type_element.value(), next_unused_index));
        next_unused_index = std::get<int>(partial_binding);
        *binding.mutable_struct_()->add_element() =
            std::get<v0::Xla::Binding>(partial_binding);
      }
      return std::make_tuple(binding, next_unused_index);
    }
    default:
      return absl::InvalidArgumentError(
          "Encountered non-tensor or struct value in attempting to construct "
          "an XLA binding.");
  }
}

inline v0::Value ComputationV(absl::optional<v0::Xla::Binding> in_binding,
                              v0::Xla::Binding out_binding,
                              xla::XlaComputation xla_comp,
                              v0::Type computation_type) {
  v0::Value value_pb;
  v0::Computation* comp_pb = value_pb.mutable_computation();
  comp_pb->mutable_xla()->mutable_hlo_module()->PackFrom(xla_comp.proto());
  *comp_pb->mutable_type() = computation_type;
  if (in_binding.has_value()) {
    *comp_pb->mutable_xla()->mutable_parameter() = in_binding.value();
  }
  *comp_pb->mutable_xla()->mutable_result() = out_binding;
  return value_pb;
}

// Creates an XLA shape via TF's TensorShapeToXLAShape with unknown rank set to
// true.
inline xla::Shape UnknownRankShapeWithDtype(tensorflow::DataType dtype) {
  tensorflow::TensorShapeProto tensor_shape;
  tensor_shape.set_unknown_rank(true);
  xla::PrimitiveType xla_type;
  tensorflow::DataTypeToPrimitiveType(dtype, &xla_type).IgnoreError();
  tensorflow::PartialTensorShape partial_shape;
  tensorflow::PartialTensorShape::BuildPartialTensorShape(tensor_shape,
                                                          &partial_shape)
      .IgnoreError();
  xla::Shape to_return =
      tensorflow::TensorShapeToXLAShape(xla_type, partial_shape);
  return to_return;
}

// Creates an XLA shape via TF's TensorShapeToXLAShape with known rank, but each
// dimension of unknown shape.
inline xla::Shape XLAShapeWithUnknownDims(tensorflow::DataType dtype,
                                          int num_dims) {
  tensorflow::TensorShapeProto tensor_shape;
  tensor_shape.set_unknown_rank(false);
  for (int i = 0; i < num_dims; i++) {
    // TensorShapeProto uses -1 to represent unknown dim, just like TFF.
    tensor_shape.add_dim()->set_size(-1);
  }
  xla::PrimitiveType xla_type;
  tensorflow::DataTypeToPrimitiveType(dtype, &xla_type).IgnoreError();
  tensorflow::PartialTensorShape partial_shape;
  tensorflow::PartialTensorShape::BuildPartialTensorShape(tensor_shape,
                                                          &partial_shape)
      .IgnoreError();
  xla::Shape to_return =
      tensorflow::TensorShapeToXLAShape(xla_type, partial_shape);
  return to_return;
}

class XLAExecutorTest : public ::testing::Test {
 public:
  XLAExecutorTest() { test_executor_ = CreateXLAExecutor("Host").value(); }
  std::shared_ptr<Executor> test_executor_;

  void CheckMaterializeEqual(ValueId id, v0::Value expected_result) {
    v0::Value output_pb;
    EXPECT_THAT(test_executor_->Materialize(id, &output_pb), IsOk());
    EXPECT_THAT(output_pb, testing::proto::IgnoringRepeatedFieldOrdering(
                               EqualsProto(expected_result)));
  }

  void CheckMaterializeStatusIs(
      ValueId id, ::testing::Matcher<absl::Status> status_matcher) {
    v0::Value output_pb;
    EXPECT_THAT(test_executor_->Materialize(id, &output_pb), status_matcher);
  }

  void CheckRoundTrip(v0::Value& input_pb) {
    TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId id,
                             test_executor_->CreateValue(input_pb));
    v0::Value output_pb;
    CheckMaterializeEqual(id, input_pb);
  }

  template <typename... Ts>
  void CheckTensorRoundTrip(Ts... tensor_constructor_args) {
    auto input_pb = TensorV(tensor_constructor_args...);
    CheckRoundTrip(input_pb);
  }

  void CheckRoundTripFails(v0::Value& input_pb,
                           ::testing::Matcher<absl::Status> status_matcher) {
    TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId id,
                             test_executor_->CreateValue(input_pb));
    v0::Value output_pb;
    EXPECT_THAT(test_executor_->Materialize(id, &output_pb), status_matcher);
  }
};

TEST_F(XLAExecutorTest, RoundTripSimpleTensor) {
  int8_t input_int = 9;
  CheckTensorRoundTrip(input_int);
}

TEST_F(XLAExecutorTest, RoundTripInt64Tensor) {
  int64_t input_int = 9;
  CheckTensorRoundTrip(input_int);
}

TEST_F(XLAExecutorTest, RoundTripFloatTensor) {
  float input_float = 9;
  CheckTensorRoundTrip(input_float);
}

TEST_F(XLAExecutorTest, RoundTripNonScalarFloatTensor) {
  tensorflow::Tensor input_tensor(tensorflow::DataType::DT_FLOAT,
                                  tensorflow::TensorShape({10, 10}));
  CheckTensorRoundTrip(input_tensor);
}

TEST_F(XLAExecutorTest, RoundTripStructWithTensor) {
  v0::Value input_pb = StructV({TensorV(9)});
  CheckRoundTrip(input_pb);
}

TEST_F(XLAExecutorTest, RoundTripStructOfNestedTensors) {
  v0::Value input_pb = StructV({StructV({TensorV(24)}), TensorV(88)});
  CheckRoundTrip(input_pb);
}

TEST_F(XLAExecutorTest, RoundTripStringTensorFails) {
  // String tensors are unsupported in XLA; see
  // https://github.com/tensorflow/tensorflow/issues/19140, and the enumeration
  // of primitive dtypes at
  // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/xla_data.proto
  auto string_tensor = TensorV("a_string");
  CheckRoundTripFails(
      string_tensor,
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Unsupported type in DataTypeToPrimitiveType: 'string'")));
}

TEST_F(XLAExecutorTest, CreateStructFailsUnimplemented) {
  v0::Value tensor_pb = TensorV(2);

  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_tensor,
                           test_executor_->CreateValue(tensor_pb));
  EXPECT_THAT(test_executor_->CreateStruct({embedded_tensor.ref()}),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(XLAExecutorTest, CreateSelectionFailsUnimplemented) {
  v0::Value tensor_pb = TensorV(2);

  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_tensor,
                           test_executor_->CreateValue(tensor_pb));
  EXPECT_THAT(test_executor_->CreateSelection(embedded_tensor.ref(), 0),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(XLAExecutorTest, CreateCallFailsUnimplemented) {
  v0::Value tensor_pb = TensorV(2);
  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_tensor,
                           test_executor_->CreateValue(tensor_pb));
  EXPECT_THAT(test_executor_->CreateCall(embedded_tensor.ref(), absl::nullopt),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(XLAExecutorTest, CreateValueComputationNonFunctionalTypeFails) {
  xla::XlaBuilder builder("float_unk_shape_tensor_identity");
  xla::Parameter(&builder, 0, xla::ShapeUtil::MakeScalarShape(xla::F32), "x");
  tensorflow::StatusOr<xla::XlaComputation> xla_computation = builder.Build();
  ASSERT_TRUE(xla_computation.ok());
  v0::Type float_tensor_type;
  float_tensor_type.mutable_tensor()->set_dtype(v0::TensorType::DT_FLOAT);

  auto tensor_binding =
      std::get<0>(TFF_ASSERT_OK(BindingFromType(float_tensor_type, 0)));
  v0::Value computation =
      ComputationV(tensor_binding, tensor_binding, std::move(*xla_computation),
                   // Create a computation with non-functional type; the XLA
                   // executor needs the functional type to porpagate shape and
                   // type information to parameters and results.
                   float_tensor_type);

  // Materialization would fail in any case; we assert here that it fails early,
  // on computation embedding.
  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_fn,
                           test_executor_->CreateValue(computation));
  CheckMaterializeStatusIs(embedded_fn,
                           StatusIs(absl::StatusCode::kInvalidArgument,
                                    HasSubstr("non-functional type")));
}

TEST_F(XLAExecutorTest, CreateValueComputationMismatchedTypeAndBindingFails) {
  xla::XlaBuilder builder("float_unk_shape_tensor_identity");
  xla::Parameter(&builder, 0, xla::ShapeUtil::MakeScalarShape(xla::F32), "x");
  tensorflow::StatusOr<xla::XlaComputation> xla_computation = builder.Build();
  ASSERT_TRUE(xla_computation.ok());
  v0::Type float_tensor;
  float_tensor.mutable_tensor()->set_dtype(v0::TensorType::DT_FLOAT);
  v0::Type function_type;
  *function_type.mutable_function()->mutable_result() = float_tensor;
  *function_type.mutable_function()->mutable_parameter() = float_tensor;
  // We create a binding with mismatched structure.
  v0::Type struct_type;
  *struct_type.mutable_struct_()->add_element()->mutable_value() = float_tensor;

  auto struct_binding =
      std::get<0>(TFF_ASSERT_OK(BindingFromType(struct_type, 0)));
  auto tensor_binding =
      std::get<0>(TFF_ASSERT_OK(BindingFromType(float_tensor, 0)));
  v0::Value computation =
      ComputationV(struct_binding, tensor_binding, std::move(*xla_computation),
                   function_type);

  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_fn,
                           test_executor_->CreateValue(computation));
  CheckMaterializeStatusIs(
      embedded_fn,
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Mismatch between tensor type and non-tensor binding")));
}

TEST_F(XLAExecutorTest,
       CreateValueComputationTensorParameterKnownRankUnknownDimsFails) {
  int num_dims = 3;
  xla::XlaBuilder builder("float_unk_shape_tensor_identity");
  xla::Parameter(&builder, 0,
                 XLAShapeWithUnknownDims(tensorflow::DT_FLOAT, num_dims), "x");
  tensorflow::StatusOr<xla::XlaComputation> xla_computation = builder.Build();
  ASSERT_TRUE(xla_computation.ok());
  v0::Type float_unk_shape_tensor;
  float_unk_shape_tensor.mutable_tensor()->set_dtype(v0::TensorType::DT_FLOAT);
  for (int i = 0; i < num_dims; i++) {
    float_unk_shape_tensor.mutable_tensor()->add_dims(-1);
  }
  v0::Type function_type;
  *function_type.mutable_function()->mutable_result() = float_unk_shape_tensor;
  *function_type.mutable_function()->mutable_parameter() =
      float_unk_shape_tensor;
  auto binding =
      std::get<0>(TFF_ASSERT_OK(BindingFromType(float_unk_shape_tensor, 0)));
  v0::Value computation = ComputationV(
      binding, binding, std::move(*xla_computation), function_type);

  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_fn,
                           test_executor_->CreateValue(computation));
  CheckMaterializeStatusIs(
      embedded_fn, StatusIs(absl::StatusCode::kInvalidArgument,
                            HasSubstr("Tensor parameters of unknown shape")));
}

TEST_F(XLAExecutorTest, CreateValueComputationTensorParameterUnknownRankFails) {
  xla::XlaBuilder builder("float_unk_rank_tensor_identity");
  xla::Parameter(&builder, 0, UnknownRankShapeWithDtype(tensorflow::DT_FLOAT),
                 "x");
  tensorflow::StatusOr<xla::XlaComputation> xla_computation = builder.Build();
  ASSERT_TRUE(xla_computation.ok());
  v0::Type float_unk_rank_tensor;
  float_unk_rank_tensor.mutable_tensor()->set_dtype(v0::TensorType::DT_FLOAT);
  float_unk_rank_tensor.mutable_tensor()->set_unknown_rank(true);
  v0::Type function_type;
  *function_type.mutable_function()->mutable_result() = float_unk_rank_tensor;
  *function_type.mutable_function()->mutable_parameter() =
      float_unk_rank_tensor;
  auto binding =
      std::get<0>(TFF_ASSERT_OK(BindingFromType(float_unk_rank_tensor, 0)));
  v0::Value computation = ComputationV(
      binding, binding, std::move(*xla_computation), function_type);

  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_fn,
                           test_executor_->CreateValue(computation));
  CheckMaterializeStatusIs(
      embedded_fn, StatusIs(absl::StatusCode::kInvalidArgument,
                            HasSubstr("Tensor parameters of unknown rank")));
}

}  // namespace
}  // namespace tensorflow_federated

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

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "federated_language/proto/array.pb.h"
#include "federated_language/proto/computation.pb.h"
#include "federated_language/proto/data_type.pb.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/stream_executor/platform.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow_federated/cc/core/impl/executors/array_shape_test_utils.h"
#include "tensorflow_federated/cc/core/impl/executors/array_test_utils.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/type_test_utils.h"
#include "tensorflow_federated/cc/core/impl/executors/value_test_utils.h"
#include "tensorflow_federated/cc/core/impl/executors/xla_utils.h"
#include "tensorflow_federated/cc/testing/protobuf_matchers.h"
#include "tensorflow_federated/cc/testing/status_matchers.h"

ABSL_FLAG(std::string, tff_xla_executor_test_platform, "Host",
          "The name of the XLA platform to run the tests on. By default will "
          "run on 'Host' (CPU).");

namespace tensorflow_federated {
namespace {

using ::tensorflow_federated::testing::EqualsProto;
using ::tensorflow_federated::testing::FlatStructT;
using ::tensorflow_federated::testing::FunctionT;
using ::tensorflow_federated::testing::IdentityFunctionT;
using ::tensorflow_federated::testing::NestedStructT;
using ::tensorflow_federated::testing::NoArgFunctionT;
using ::tensorflow_federated::testing::StructT;
using ::tensorflow_federated::testing::StructV;
using ::tensorflow_federated::testing::TensorT;
using ::testing::HasSubstr;

inline absl::StatusOr<std::tuple<federated_language::Xla::Binding, int>>
BindingFromType(federated_language::Type type, int next_unused_index) {
  switch (type.type_case()) {
    case federated_language::Type::kTensor: {
      federated_language::Xla::Binding binding;
      binding.mutable_tensor()->set_index(next_unused_index);
      return std::make_tuple(binding, next_unused_index + 1);
    }
    case federated_language::Type::kStruct: {
      federated_language::Xla::Binding binding;
      for (const auto& type_element : type.struct_().element()) {
        auto partial_binding =
            TFF_TRY(BindingFromType(type_element.value(), next_unused_index));
        next_unused_index = std::get<int>(partial_binding);
        *binding.mutable_struct_()->add_element() =
            std::get<federated_language::Xla::Binding>(partial_binding);
      }
      return std::make_tuple(binding, next_unused_index);
    }
    default:
      return absl::InvalidArgumentError(
          "Encountered non-tensor or struct value in attempting to construct "
          "an XLA binding.");
  }
}

inline v0::Value ComputationV(
    std::optional<federated_language::Xla::Binding> in_binding,
    federated_language::Xla::Binding out_binding, xla::XlaComputation xla_comp,
    federated_language::Type computation_type) {
  v0::Value value_pb;
  federated_language::Computation* comp_pb = value_pb.mutable_computation();
  comp_pb->mutable_xla()->mutable_hlo_module()->PackFrom(xla_comp.proto());
  *comp_pb->mutable_type() = computation_type;
  if (in_binding.has_value()) {
    *comp_pb->mutable_xla()->mutable_parameter() = in_binding.value();
  }
  *comp_pb->mutable_xla()->mutable_result() = out_binding;
  return value_pb;
}

// Creates an XLA shape with unknown rank.
absl::StatusOr<xla::Shape> ShapeWithUnknownRank(
    federated_language::DataType data_type) {
  xla::PrimitiveType element_type =
      TFF_TRY(PrimitiveTypeFromDataType(data_type));
  // For unknown rank, create a rank 1 size 0 tensor.
  return xla::ShapeUtil::MakeShapeWithDenseLayout(element_type, {0}, {0});
}

// Creates an XLA shape with known rank, but each dimension has unknown shape.
absl::StatusOr<xla::Shape> ShapeWithUnknownDims(
    federated_language::DataType data_type, int num_dims) {
  xla::PrimitiveType element_type =
      TFF_TRY(PrimitiveTypeFromDataType(data_type));
  std::vector<int64_t> dimensions(num_dims, 0);
  return xla::ShapeUtil::MakeShape(element_type, dimensions);
}

class XLAExecutorTest : public ::testing::Test {
 public:
  XLAExecutorTest() {
    absl::StatusOr<std::vector<xla::se::Platform*>> platforms =
        xla::PlatformUtil::GetSupportedPlatforms();
    if (!platforms.ok()) {
      LOG(FATAL) << "Could not enumerate supported XLA platforms";
    }
    LOG(INFO) << "Found " << platforms->size() << " platforms";
    const std::string requested_platform_name =
        absl::GetFlag(FLAGS_tff_xla_executor_test_platform);
    for (auto* platform : *platforms) {
      LOG(INFO) << "Platform: " << platform->Name();
      if (platform->Name() == requested_platform_name) {
        if (platform->VisibleDeviceCount() > 0) {
          test_executor_ = CreateXLAExecutor(platform->Name()).value();
        }
      }
    }
    // Fail the test if we couldn't find the requested platform.
    CHECK(test_executor_ != nullptr)
        << "Could not find platform " << requested_platform_name
        << ", missing build dependency, or no devices for platform found. See "
        << "logs for registered platforms.";
  }
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

  void CheckRoundTripFails(v0::Value& input_pb,
                           ::testing::Matcher<absl::Status> status_matcher) {
    TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId id,
                             test_executor_->CreateValue(input_pb));
    v0::Value output_pb;
    EXPECT_THAT(test_executor_->Materialize(id, &output_pb), status_matcher);
  }
};

TEST_F(XLAExecutorTest, RoundTripSimpleTensor) {
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT8,
                                         testing::CreateArrayShape({}), {1}));
  v0::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  CheckRoundTrip(value_pb);
}

TEST_F(XLAExecutorTest, RoundTripInt64Tensor) {
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT64,
                                         testing::CreateArrayShape({}), {1}));
  v0::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  CheckRoundTrip(value_pb);
}

TEST_F(XLAExecutorTest, RoundTripFloatTensor) {
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  CheckRoundTrip(value_pb);
}

TEST_F(XLAExecutorTest, RoundTripNonScalarFloatTensor) {
  federated_language::Array array_pb = TFF_ASSERT_OK(
      testing::CreateArray(federated_language::DataType::DT_FLOAT,
                           testing::CreateArrayShape({2}), {1.0, 1.0}));
  v0::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  CheckRoundTrip(value_pb);
}

TEST_F(XLAExecutorTest, RoundTripStructWithTensor) {
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT64,
                                         testing::CreateArrayShape({}), {1}));
  v0::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  v0::Value input_pb = StructV({value_pb});
  CheckRoundTrip(input_pb);
}

TEST_F(XLAExecutorTest, RoundTripStructOfNestedTensors) {
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  v0::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {2}));
  v0::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  v0::Value input_pb = StructV({StructV({value1_pb}), value2_pb});
  CheckRoundTrip(input_pb);
}

TEST_F(XLAExecutorTest, RoundTripStringTensorFails) {
  federated_language::Array array_pb = TFF_ASSERT_OK(
      testing::CreateArray(federated_language::DataType::DT_STRING,
                           testing::CreateArrayShape({}), {"a"}));
  v0::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  CheckRoundTripFails(value_pb, StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(XLAExecutorTest, CreateStructOneElement) {
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  v0::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  TFF_ASSERT_OK_AND_ASSIGN(auto value, test_executor_->CreateValue(value_pb));
  TFF_ASSERT_OK_AND_ASSIGN(auto struct_, test_executor_->CreateStruct({value}));
  CheckMaterializeEqual(struct_, StructV({value_pb}));
}

TEST_F(XLAExecutorTest, CreateStructSeveralElements) {
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  v0::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {2}));
  v0::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  federated_language::Array array3_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {3}));
  v0::Value value3_pb;
  value3_pb.mutable_array()->Swap(&array3_pb);
  v0::Value struct_ = StructV({value1_pb, value2_pb, value3_pb});
  TFF_ASSERT_OK_AND_ASSIGN(auto t1id, test_executor_->CreateValue(value1_pb));
  TFF_ASSERT_OK_AND_ASSIGN(auto t2id, test_executor_->CreateValue(value2_pb));
  TFF_ASSERT_OK_AND_ASSIGN(auto t3id, test_executor_->CreateValue(value3_pb));
  TFF_ASSERT_OK_AND_ASSIGN(auto structid,
                           test_executor_->CreateStruct({t1id, t2id, t3id}));
  CheckMaterializeEqual(structid, struct_);
}

TEST_F(XLAExecutorTest, CreateSelectionFromCreateValue) {
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  v0::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {2}));
  v0::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  v0::Value input = StructV({value1_pb, value2_pb});
  TFF_ASSERT_OK_AND_ASSIGN(auto vid, test_executor_->CreateValue(input));
  TFF_ASSERT_OK_AND_ASSIGN(auto t1id, test_executor_->CreateSelection(vid, 1));
  CheckMaterializeEqual(t1id, value2_pb);
}

TEST_F(XLAExecutorTest, CreateSelectionFromCreateStruct) {
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  v0::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {2}));
  v0::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  v0::Value input = StructV({value1_pb, value2_pb});
  TFF_ASSERT_OK_AND_ASSIGN(auto t1id, test_executor_->CreateValue(value1_pb));
  TFF_ASSERT_OK_AND_ASSIGN(auto t2id, test_executor_->CreateValue(value2_pb));
  TFF_ASSERT_OK_AND_ASSIGN(auto structid,
                           test_executor_->CreateStruct({t1id, t2id}));
  TFF_ASSERT_OK_AND_ASSIGN(auto selectedid,
                           test_executor_->CreateSelection(structid, 1));
  CheckMaterializeEqual(selectedid, value2_pb);
}

TEST_F(XLAExecutorTest, CreateSelectionNonStructImmediate) {
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_INT32,
                                         testing::CreateArrayShape({}), {1}));
  v0::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  TFF_ASSERT_OK_AND_ASSIGN(auto id, test_executor_->CreateValue(value1_pb));
  CheckMaterializeEqual(id, value1_pb);
  EXPECT_THAT(
      test_executor_->CreateSelection(id, 0),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Cannot create selection on non-struct value.")));
}

TEST_F(XLAExecutorTest, CreateSelectionOOBImmediate) {
  TFF_ASSERT_OK_AND_ASSIGN(auto id, test_executor_->CreateValue(StructV({})));
  CheckMaterializeEqual(id, StructV({}));
  EXPECT_THAT(
      test_executor_->CreateSelection(id, 0),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Attempted to access index 0 of a 0-length struct.")));
}

TEST_F(XLAExecutorTest, CreateValueComputationTensorNonFunctionalTypeFails) {
  xla::XlaBuilder builder("float_unk_shape_tensor_identity");
  xla::Parameter(&builder, 0, xla::ShapeUtil::MakeScalarShape(xla::F32), "x");
  absl::StatusOr<xla::XlaComputation> xla_computation = builder.Build();
  ASSERT_TRUE(xla_computation.ok());
  federated_language::Type float_tensor_type;
  float_tensor_type.mutable_tensor()->set_dtype(
      federated_language::DataType::DT_FLOAT);

  auto tensor_binding =
      std::get<0>(TFF_ASSERT_OK(BindingFromType(float_tensor_type, 0)));
  v0::Value computation =
      ComputationV(tensor_binding, tensor_binding, std::move(*xla_computation),
                   // Create a computation with non-functional type; the XLA
                   // executor needs the functional type to propagate shape and
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

TEST_F(XLAExecutorTest,
       CreateValueComputationTensorMismatchedTypeAndBindingFails) {
  xla::XlaBuilder builder("float_unk_shape_tensor_identity");
  xla::Parameter(&builder, 0, xla::ShapeUtil::MakeScalarShape(xla::F32), "x");
  absl::StatusOr<xla::XlaComputation> xla_computation = builder.Build();
  ASSERT_TRUE(xla_computation.ok());
  federated_language::Type float_tensor;
  float_tensor.mutable_tensor()->set_dtype(
      federated_language::DataType::DT_FLOAT);
  federated_language::Type function_type;
  *function_type.mutable_function()->mutable_result() = float_tensor;
  *function_type.mutable_function()->mutable_parameter() = float_tensor;
  // We create a binding with mismatched structure.
  federated_language::Type struct_type;
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
  xla::Shape shape = TFF_ASSERT_OK(
      ShapeWithUnknownDims(federated_language::DataType::DT_FLOAT, num_dims));
  xla::Parameter(&builder, 0, shape, "x");
  absl::StatusOr<xla::XlaComputation> xla_computation = builder.Build();
  ASSERT_TRUE(xla_computation.ok());
  federated_language::Type float_unk_shape_tensor;
  float_unk_shape_tensor.mutable_tensor()->set_dtype(
      federated_language::DataType::DT_FLOAT);
  for (int i = 0; i < num_dims; i++) {
    float_unk_shape_tensor.mutable_tensor()->add_dims(-1);
  }
  federated_language::Type function_type;
  *function_type.mutable_function()->mutable_result() = float_unk_shape_tensor;
  *function_type.mutable_function()->mutable_parameter() =
      float_unk_shape_tensor;
  auto binding =
      std::get<0>(TFF_ASSERT_OK(BindingFromType(float_unk_shape_tensor, 0)));
  v0::Value computation = ComputationV(
      binding, binding, std::move(*xla_computation), function_type);

  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_fn,
                           test_executor_->CreateValue(computation));
  CheckMaterializeStatusIs(embedded_fn,
                           StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(XLAExecutorTest, CreateValueComputationTensorParameterUnknownRankFails) {
  xla::XlaBuilder builder("float_unk_rank_tensor_identity");
  xla::Shape shape = TFF_ASSERT_OK(
      ShapeWithUnknownRank(federated_language::DataType::DT_FLOAT));
  xla::Parameter(&builder, 0, shape, "x");
  absl::StatusOr<xla::XlaComputation> xla_computation = builder.Build();
  ASSERT_TRUE(xla_computation.ok());
  federated_language::Type float_unk_rank_tensor;
  float_unk_rank_tensor.mutable_tensor()->set_dtype(
      federated_language::DataType::DT_FLOAT);
  float_unk_rank_tensor.mutable_tensor()->set_unknown_rank(true);
  federated_language::Type function_type;
  *function_type.mutable_function()->mutable_result() = float_unk_rank_tensor;
  *function_type.mutable_function()->mutable_parameter() =
      float_unk_rank_tensor;
  auto binding =
      std::get<0>(TFF_ASSERT_OK(BindingFromType(float_unk_rank_tensor, 0)));
  v0::Value computation = ComputationV(
      binding, binding, std::move(*xla_computation), function_type);

  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_fn,
                           test_executor_->CreateValue(computation));
  CheckMaterializeStatusIs(embedded_fn,
                           StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(XLAExecutorTest, CreateValueComputationLiteralReturnsResult) {
  const federated_language::DataType dtype =
      federated_language::DataType::DT_INT32;
  federated_language::ArrayShape shape_pb = testing::CreateArrayShape({});
  auto values = {1};
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(dtype, shape_pb, values));
  federated_language::Computation computation_pb =
      testing::LiteralComputation(array_pb);
  v0::Value value_pb = testing::ComputationV(computation_pb);

  const OwnedValueId& embedded_fn =
      TFF_ASSERT_OK(test_executor_->CreateValue(value_pb));

  v0::Value expected_pb;
  expected_pb.mutable_array()->Swap(&array_pb);
  CheckMaterializeEqual(embedded_fn, expected_pb);
}

TEST_F(XLAExecutorTest, CreateAndMaterializeNoArgCallSingleTensor) {
  xla::XlaBuilder builder("return_one");
  xla::XlaOp constant = xla::ConstantR0<float>(&builder, 1.0);
  // To mimic the Python tracing which always returns tuples, event for single
  // element results, after passing through MLIR
  // results are always in tuples.
  xla::Tuple(&builder, {constant});
  absl::StatusOr<xla::XlaComputation> xla_computation = builder.Build();
  ASSERT_TRUE(xla_computation.ok());
  auto tensor_type = TensorT(federated_language::DataType::DT_FLOAT);
  federated_language::Type function_type = NoArgFunctionT(tensor_type);
  v0::Value computation = ComputationV(
      std::nullopt, std::get<0>(TFF_ASSERT_OK(BindingFromType(tensor_type, 0))),
      std::move(*xla_computation), function_type);

  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_fn,
                           test_executor_->CreateValue(computation));
  TFF_ASSERT_OK_AND_ASSIGN(
      OwnedValueId called_fn,
      test_executor_->CreateCall(embedded_fn.ref(), std::nullopt));
  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value expected_pb;
  expected_pb.mutable_array()->Swap(&array_pb);
  CheckMaterializeEqual(called_fn, expected_pb);
}

TEST_F(XLAExecutorTest, CreateAndMaterializeNoArgCallTensorStructure) {
  xla::XlaBuilder builder("return_two_tensors");
  auto float_one = xla::ConstantR0<float>(&builder, 1.0);
  auto float_two = xla::ConstantR0<float>(&builder, 2.0);
  xla::Tuple(&builder, {float_one, float_two});
  absl::StatusOr<xla::XlaComputation> xla_computation = builder.Build();
  ASSERT_TRUE(xla_computation.ok());

  federated_language::Type return_type =
      FlatStructT(federated_language::DataType::DT_FLOAT, 2);
  federated_language::Type function_type = NoArgFunctionT(return_type);

  v0::Value computation = ComputationV(
      std::nullopt, std::get<0>(TFF_ASSERT_OK(BindingFromType(return_type, 0))),
      std::move(*xla_computation), function_type);

  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_fn,
                           test_executor_->CreateValue(computation));
  TFF_ASSERT_OK_AND_ASSIGN(
      OwnedValueId called_fn,
      test_executor_->CreateCall(embedded_fn.ref(), std::nullopt));
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {2.0}));
  v0::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  v0::Value expected_result = StructV({value1_pb, value2_pb});
  CheckMaterializeEqual(called_fn, expected_result);
}

TEST_F(XLAExecutorTest, CreateAndMaterializeNoArgCallNestedTensorStructure) {
  xla::XlaBuilder builder("return_nested_struct");
  auto float_one = xla::ConstantR0<float>(&builder, 1.0);
  auto float_two = xla::ConstantR0<float>(&builder, 2.0);
  auto float_three = xla::ConstantR0<float>(&builder, 3.0);
  xla::Tuple(&builder, {float_one, float_two, float_three});
  absl::StatusOr<xla::XlaComputation> xla_computation = builder.Build();
  ASSERT_TRUE(xla_computation.ok());

  // We construct a return type <tf.float32, <tf.float32, tf.float32>>
  federated_language::Type nested_struct_type =
      NestedStructT(federated_language::DataType::DT_FLOAT);
  federated_language::Type function_type = NoArgFunctionT(nested_struct_type);

  v0::Value computation = ComputationV(
      std::nullopt,
      std::get<0>(TFF_ASSERT_OK(BindingFromType(nested_struct_type, 0))),
      std::move(*xla_computation), function_type);

  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_fn,
                           test_executor_->CreateValue(computation));
  TFF_ASSERT_OK_AND_ASSIGN(
      OwnedValueId called_fn,
      test_executor_->CreateCall(embedded_fn.ref(), std::nullopt));
  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {2.0}));
  v0::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  federated_language::Array array3_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {3.0}));
  v0::Value value3_pb;
  value3_pb.mutable_array()->Swap(&array3_pb);
  v0::Value expected_result =
      StructV({value1_pb, StructV({value2_pb, value3_pb})});

  CheckMaterializeEqual(called_fn, expected_result);
}

TEST_F(XLAExecutorTest, CreateAndMaterializeIdentityScalar) {
  xla::XlaBuilder builder("float_scalar_identity");
  xla::XlaOp parameter = xla::Parameter(
      &builder, 0, xla::ShapeUtil::MakeScalarShape(xla::F32), "x");
  // To mimic the Python tracing which always returns tuples, event for single
  // element results, after passing through MLIR results are always in tuples.
  xla::Tuple(&builder, {parameter});
  absl::StatusOr<xla::XlaComputation> xla_computation = builder.Build();
  ASSERT_TRUE(xla_computation.ok());
  federated_language::Type float_tensor_type =
      TensorT(federated_language::DataType::DT_FLOAT);
  federated_language::Type function_type = IdentityFunctionT(float_tensor_type);
  auto binding =
      std::get<0>(TFF_ASSERT_OK(BindingFromType(float_tensor_type, 0)));
  v0::Value computation = ComputationV(
      // Identical parameter and result bindings.
      binding, binding, std::move(*xla_computation), function_type);

  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_fn,
                           test_executor_->CreateValue(computation));
  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_arg,
                           test_executor_->CreateValue(value_pb));
  TFF_ASSERT_OK_AND_ASSIGN(
      OwnedValueId called_fn,
      test_executor_->CreateCall(embedded_fn.ref(), embedded_arg));
  CheckMaterializeEqual(called_fn, value_pb);
}

TEST_F(XLAExecutorTest, CreateAndMaterializeIdentitySingletonStruct) {
  xla::XlaBuilder builder("float_scalar_singleton_struct");
  xla::XlaOp parameter = xla::Parameter(
      &builder, 0, xla::ShapeUtil::MakeScalarShape(xla::F32), "x");
  // To mimic the Python tracing which always returns tuples, event for single
  // element results, after passing through MLIR results are always in tuples.
  xla::Tuple(&builder, {parameter});
  absl::StatusOr<xla::XlaComputation> xla_computation = builder.Build();
  ASSERT_TRUE(xla_computation.ok());
  federated_language::Type single_float_struct_type =
      StructT({TensorT(federated_language::DataType::DT_FLOAT)});
  federated_language::Type function_type =
      IdentityFunctionT(single_float_struct_type);
  auto binding =
      std::get<0>(TFF_ASSERT_OK(BindingFromType(single_float_struct_type, 0)));
  v0::Value computation = ComputationV(
      // Identical parameter and result bindings.
      binding, binding, std::move(*xla_computation), function_type);

  federated_language::Array array_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value value_pb;
  value_pb.mutable_array()->Swap(&array_pb);
  v0::Value arg_value = StructV({value_pb});
  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_fn,
                           test_executor_->CreateValue(computation));
  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_arg,
                           test_executor_->CreateValue(arg_value));
  TFF_ASSERT_OK_AND_ASSIGN(
      OwnedValueId called_fn,
      test_executor_->CreateCall(embedded_fn.ref(), embedded_arg));
  CheckMaterializeEqual(called_fn, arg_value);
}

TEST_F(XLAExecutorTest, CreateAndMaterializeIdentityNestedStruct) {
  xla::XlaBuilder builder("float_nested_struct_identity");
  auto x = xla::Parameter(&builder, 0,
                          xla::ShapeUtil::MakeScalarShape(xla::F32), "x");
  auto y = xla::Parameter(&builder, 1,
                          xla::ShapeUtil::MakeScalarShape(xla::F32), "y");
  auto z = xla::Parameter(&builder, 2,
                          xla::ShapeUtil::MakeScalarShape(xla::F32), "z");
  xla::Tuple(&builder, {x, y, z});
  absl::StatusOr<xla::XlaComputation> xla_computation = builder.Build();
  ASSERT_TRUE(xla_computation.ok());

  federated_language::Type nested_struct_type =
      NestedStructT(federated_language::DataType::DT_FLOAT);
  federated_language::Type function_type =
      IdentityFunctionT(nested_struct_type);
  auto binding =
      std::get<0>(TFF_ASSERT_OK(BindingFromType(nested_struct_type, 0)));
  v0::Value computation = ComputationV(
      binding, binding, std::move(*xla_computation), function_type);

  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {2.0}));
  v0::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  federated_language::Array array3_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {3.0}));
  v0::Value value3_pb;
  value3_pb.mutable_array()->Swap(&array3_pb);
  v0::Value arg_value = StructV({value1_pb, StructV({value2_pb, value3_pb})});
  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_fn,
                           test_executor_->CreateValue(computation));
  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_arg,
                           test_executor_->CreateValue(arg_value));
  TFF_ASSERT_OK_AND_ASSIGN(
      OwnedValueId called_fn,
      test_executor_->CreateCall(embedded_fn.ref(), embedded_arg));
  CheckMaterializeEqual(called_fn, arg_value);
}

TEST_F(XLAExecutorTest, CallAndMaterializeIdentityPartiallyNonScalarStruct) {
  xla::XlaBuilder builder("partially_non_scalar_struct_identity");
  auto x = xla::Parameter(&builder, 0,
                          xla::ShapeUtil::MakeScalarShape(xla::F32), "x");
  auto y = xla::Parameter(&builder, 1,
                          xla::ShapeUtil::MakeShape(xla::F32, {2, 3}), "y");
  xla::Tuple(&builder, {x, y});
  absl::StatusOr<xla::XlaComputation> xla_computation = builder.Build();
  ASSERT_TRUE(xla_computation.ok());

  // Create a computation type to match the above.
  federated_language::Type scalar =
      TensorT(federated_language::DataType::DT_FLOAT);
  federated_language::Type matrix =
      TensorT(federated_language::DataType::DT_FLOAT, {2, 3});
  federated_language::Type struct_type = StructT({scalar, matrix});
  federated_language::Type function_type = IdentityFunctionT(struct_type);
  auto binding = std::get<0>(TFF_ASSERT_OK(BindingFromType(struct_type, 0)));
  v0::Value computation = ComputationV(
      binding, binding, std::move(*xla_computation), function_type);

  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  federated_language::Array array2_pb = TFF_ASSERT_OK(testing::CreateArray(
      federated_language::DataType::DT_FLOAT, testing::CreateArrayShape({2, 3}),
      {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}));
  v0::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  v0::Value arg_value = StructV({value1_pb, value2_pb});
  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_fn,
                           test_executor_->CreateValue(computation));
  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_arg,
                           test_executor_->CreateValue(arg_value));
  TFF_ASSERT_OK_AND_ASSIGN(
      OwnedValueId called_fn,
      test_executor_->CreateCall(embedded_fn.ref(), embedded_arg));
  CheckMaterializeEqual(called_fn, arg_value);
}

TEST_F(XLAExecutorTest,
       CreateCallAndMaterializeDifferentParameterAndResultTypes) {
  xla::XlaBuilder builder("float_nested_struct_partial_sum");
  auto x = xla::Parameter(&builder, 0,
                          xla::ShapeUtil::MakeScalarShape(xla::F32), "x");
  auto y = xla::Parameter(&builder, 1,
                          xla::ShapeUtil::MakeScalarShape(xla::F32), "y");
  auto z = xla::Parameter(&builder, 2,
                          xla::ShapeUtil::MakeScalarShape(xla::F32), "z");
  xla::Tuple(&builder, {x, xla::Add(y, z)});
  absl::StatusOr<xla::XlaComputation> xla_computation = builder.Build();
  ASSERT_TRUE(xla_computation.ok());
  federated_language::Type nested_struct_type =
      NestedStructT(federated_language::DataType::DT_FLOAT);
  federated_language::Type result_type =
      FlatStructT(federated_language::DataType::DT_FLOAT, 2);
  federated_language::Type function_type =
      FunctionT(nested_struct_type, result_type);
  auto parameter_binding =
      std::get<0>(TFF_ASSERT_OK(BindingFromType(nested_struct_type, 0)));
  auto result_binding =
      std::get<0>(TFF_ASSERT_OK(BindingFromType(result_type, 0)));
  v0::Value computation =
      ComputationV(parameter_binding, result_binding,
                   std::move(*xla_computation), function_type);

  federated_language::Array array1_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {1.0}));
  v0::Value value1_pb;
  value1_pb.mutable_array()->Swap(&array1_pb);
  federated_language::Array array2_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {2.0}));
  v0::Value value2_pb;
  value2_pb.mutable_array()->Swap(&array2_pb);
  federated_language::Array array3_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {3.0}));
  v0::Value value3_pb;
  value3_pb.mutable_array()->Swap(&array3_pb);
  v0::Value arg_value = StructV({value1_pb, StructV({value2_pb, value3_pb})});
  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_fn,
                           test_executor_->CreateValue(computation));
  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId embedded_arg,
                           test_executor_->CreateValue(arg_value));
  TFF_ASSERT_OK_AND_ASSIGN(
      OwnedValueId called_fn,
      test_executor_->CreateCall(embedded_fn.ref(), embedded_arg));
  federated_language::Array array5_pb =
      TFF_ASSERT_OK(testing::CreateArray(federated_language::DataType::DT_FLOAT,
                                         testing::CreateArrayShape({}), {5.0}));
  v0::Value value5_pb;
  value5_pb.mutable_array()->Swap(&array5_pb);
  v0::Value expected_result = StructV({value1_pb, value5_pb});
  CheckMaterializeEqual(called_fn, expected_result);
}

}  // namespace
}  // namespace tensorflow_federated

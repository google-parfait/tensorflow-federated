/* Copyright 2021, The TensorFlow Federated Authors.

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

#include "tensorflow_federated/cc/core/impl/executors/tensorflow_executor.h"

#include <fcntl.h>

#include <cstdint>
#include <future>  // NOLINT
#include <memory>
#include <string>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/protobuf_matchers.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"
#include "tensorflow_federated/cc/core/impl/executors/value_test_utils.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

ABSL_FLAG(std::string, reduce_graph_path, "",
          "Path to a serialized GraphDef containing a dataset reduce.");

namespace tensorflow_federated {
namespace {

using testing::CreateSerializedRangeDatasetGraphDef;
using testing::EqualsProto;
using testing::SequenceV;
using testing::StructV;
using testing::TensorV;

template <class TfOp>
inline v0::TensorFlow::Binding TensorB(const TfOp& op) {
  const tensorflow::Node* node = op.node();
  v0::TensorFlow::Binding binding;
  *binding.mutable_tensor()->mutable_tensor_name() = node->name();
  return binding;
}

template <class TfOp>
inline v0::TensorFlow::Binding SequenceB(const TfOp& op) {
  const tensorflow::Node* node = op.node();
  v0::TensorFlow::Binding binding;
  *binding.mutable_sequence()->mutable_variant_tensor_name() = node->name();
  return binding;
}

inline v0::TensorFlow::Binding StructB(
    const absl::Span<const v0::TensorFlow::Binding> elements) {
  v0::TensorFlow::Binding binding;
  auto struct_mut = binding.mutable_struct_();
  for (const auto& element : elements) {
    *struct_mut->add_element() = element;
  }
  return binding;
}

inline v0::Value ComputationV(
    absl::optional<v0::TensorFlow::Binding> in_binding,
    v0::TensorFlow::Binding out_binding, const tensorflow::Scope& scope,
    const absl::optional<const tensorflow::Operation>& init_op =
        absl::nullopt) {
  v0::Value value_pb;
  v0::Computation* comp_pb = value_pb.mutable_computation();
  // NOTE: we do not fill in the `type` field of `comp` because it is not needed
  // by the C++ TensorFlow executor.
  v0::TensorFlow* tensorflow_pb = comp_pb->mutable_tensorflow();
  tensorflow::GraphDef graphdef_pb;
  tensorflow::Status status = scope.ToGraphDef(&graphdef_pb);
  CHECK(status.ok()) << status.error_message();
  tensorflow_pb->mutable_graph_def()->PackFrom(graphdef_pb);
  if (in_binding.has_value()) {
    *tensorflow_pb->mutable_parameter() = in_binding.value();
  }
  *tensorflow_pb->mutable_result() = out_binding;
  if (init_op.has_value()) {
    *tensorflow_pb->mutable_initialize_op() = init_op.value().node()->name();
  }
  return value_pb;
}

class TensorFlowExecutorTest : public ::testing::Test {
 public:
  TensorFlowExecutorTest() { test_executor_ = CreateTensorFlowExecutor(); }
  std::shared_ptr<Executor> test_executor_;
  void CheckRoundTrip(v0::Value& input_pb) {
    TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId id,
                             test_executor_->CreateValue(input_pb));
    v0::Value output_pb;
    EXPECT_THAT(test_executor_->Materialize(id, &output_pb), IsOk());
    EXPECT_THAT(output_pb, testing::proto::IgnoringRepeatedFieldOrdering(
                               EqualsProto(input_pb)));
  }

  template <typename... Ts>
  void CheckTensorRoundTrip(Ts... tensor_constructor_args) {
    v0::Value input_pb = TensorV(tensor_constructor_args...);
    CheckRoundTrip(input_pb);
  }

  void CheckCallEqualsProto(const v0::Value& fn,
                            const absl::optional<v0::Value>& arg,
                            const v0::Value& expected) {
    TFF_ASSERT_OK_AND_ASSIGN(auto fn_id, test_executor_->CreateValue(fn));
    absl::optional<OwnedValueId> arg_id;
    if (arg.has_value()) {
      TFF_ASSERT_OK_AND_ASSIGN(arg_id,
                               test_executor_->CreateValue(arg.value()));
    }
    TFF_ASSERT_OK_AND_ASSIGN(auto result,
                             test_executor_->CreateCall(fn_id, arg_id));
    TFF_ASSERT_OK_AND_ASSIGN(auto result_proto,
                             test_executor_->Materialize(result));
    EXPECT_THAT(result_proto, EqualsProto(expected));
  }

  void CheckCallRepeatedlyEqualsProto(const v0::Value& fn,
                                      const absl::optional<v0::Value>& arg,
                                      const v0::Value& expected) {
    TFF_ASSERT_OK_AND_ASSIGN(auto fn_id, test_executor_->CreateValue(fn));
    absl::optional<OwnedValueId> arg_id;
    if (arg.has_value()) {
      TFF_ASSERT_OK_AND_ASSIGN(arg_id,
                               test_executor_->CreateValue(arg.value()));
    }
    for (int i = 0; i < 3; i++) {
      TFF_ASSERT_OK_AND_ASSIGN(auto result,
                               test_executor_->CreateCall(fn_id, arg_id));
      TFF_ASSERT_OK_AND_ASSIGN(auto result_proto,
                               test_executor_->Materialize(result));
      EXPECT_THAT(result_proto, EqualsProto(expected));
    }
  }

  void CheckCallParallelEqualsProto(const v0::Value& fn,
                                    const absl::optional<v0::Value>& arg,
                                    const v0::Value& expected) {
    TFF_ASSERT_OK_AND_ASSIGN(auto fn_id, test_executor_->CreateValue(fn));
    absl::optional<OwnedValueId> arg_id;
    if (arg.has_value()) {
      TFF_ASSERT_OK_AND_ASSIGN(arg_id,
                               test_executor_->CreateValue(arg.value()));
    }
    const int NUM_TEST_THREADS = 32;
    std::vector<std::future<absl::StatusOr<v0::Value>>> results;
    for (int i = 0; i < NUM_TEST_THREADS; i++) {
      results.emplace_back(
          std::async(std::launch::async, [&]() -> absl::StatusOr<v0::Value> {
            return test_executor_->Materialize(
                TFF_TRY(test_executor_->CreateCall(fn_id, arg_id)));
          }));
    }
    for (std::future<absl::StatusOr<v0::Value>>& result_future : results) {
      result_future.wait();
      auto result_status = result_future.get();
      TFF_EXPECT_OK(result_status);
      if (result_status.ok()) {
        EXPECT_THAT(result_status.value(), EqualsProto(expected));
      }
    }
  }
};

TEST_F(TensorFlowExecutorTest, CreateValueEmptyStruct) {
  v0::Value input_pb;
  input_pb.mutable_struct_();
  EXPECT_THAT(test_executor_->CreateValue(input_pb), IsOk());
}

TEST_F(TensorFlowExecutorTest, CreateValueSimpleTensor) {
  int8_t input_int = 9;
  v0::Value input_pb = TensorV(input_int);
  EXPECT_THAT(test_executor_->CreateValue(input_pb), IsOk());
}

// Returns a `GraphDef` whose serialized form is stored in a file at `path`.
tensorflow::GraphDef LoadGraph(const char* path) {
  int fd = open(path, O_RDONLY);
  CHECK(fd != -1) << "Failed to open graph at path " << path;
  google::protobuf::io::FileInputStream fs(fd);
  fs.SetCloseOnDelete(true);

  tensorflow::GraphDef graph;
  bool parsed = google::protobuf::TextFormat::Parse(&fs, &graph);
  CHECK(parsed) << "Input graph at path \"" << path
                << "\" is invalid text-format GraphDef";
  return graph;
}

// Constants necessary for using the dataset-reducing graph created by
// `make_reduce_lambda_test_graph.py`.
char const* const kReduceDatasetInputName = "input_dataset_placeholder";
char const* const kReduceResultOutputTensorName = "result_tensor";

// Returns a value containing a computation which will perform a reduce on an
// input dataset of `int64_t`s and return the sum of the elements.
v0::Value CreateDatasetReduceComputationV() {
  v0::Value value_pb;
  v0::Computation* comp_pb = value_pb.mutable_computation();
  // NOTE: we do not fill in the `type` field of `comp` because it is not needed
  // by the C++ TensorFlow executor.
  v0::TensorFlow* tensorflow_pb = comp_pb->mutable_tensorflow();
  std::string reduce_graph_path = absl::GetFlag(FLAGS_reduce_graph_path);
  tensorflow::GraphDef graphdef_pb = LoadGraph(reduce_graph_path.c_str());
  tensorflow_pb->mutable_graph_def()->PackFrom(graphdef_pb);
  *tensorflow_pb->mutable_parameter()
       ->mutable_sequence()
       ->mutable_variant_tensor_name() = kReduceDatasetInputName;
  *tensorflow_pb->mutable_result()->mutable_tensor()->mutable_tensor_name() =
      kReduceResultOutputTensorName;
  return value_pb;
}

TEST_F(TensorFlowExecutorTest, CallReduceOnSequence) {
  int64_t start = 0;
  int64_t stop = 10;
  int64_t step = 2;
  int64_t expected_sum = 0 + 2 + 4 + 6 + 8;
  CheckCallEqualsProto(CreateDatasetReduceComputationV(),
                       SequenceV(start, stop, step), TensorV(expected_sum));
}

TEST_F(TensorFlowExecutorTest, RoundTripEmptyStruct) {
  v0::Value input_pb;
  input_pb.mutable_struct_();
  CheckRoundTrip(input_pb);
}

TEST_F(TensorFlowExecutorTest, RoundTripSimpleTensor) {
  int8_t an_int = 9;
  CheckTensorRoundTrip(an_int);
  float a_float = 1.0;
  CheckTensorRoundTrip(a_float);
  const char* a_string = "fooey";
  CheckTensorRoundTrip(a_string);
}

TEST_F(TensorFlowExecutorTest, RoundTripStructWithTensor) {
  v0::Value input_pb = StructV({TensorV(9)});
  CheckRoundTrip(input_pb);
}

TEST_F(TensorFlowExecutorTest, RoundTripStructOfNestedTensors) {
  v0::Value input_pb = StructV({StructV({TensorV(24)}), TensorV(88)});
  CheckRoundTrip(input_pb);
}

TEST_F(TensorFlowExecutorTest, RoundTripSequence) {
  tensorflow::tstring graph_def = CreateSerializedRangeDatasetGraphDef(
      /*start=*/0, /*stop=*/10, /*step=*/1);
  v0::Value value_pb;
  v0::Value::Sequence* sequence_pb = value_pb.mutable_sequence();
  *sequence_pb->mutable_serialized_graph_def() =
      std::string(graph_def.data(), graph_def.size());
  // We can't simply `CheckRoundTrip` because the serialized graph defs
  // don't have deterministic node orders.
  TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId id,
                           test_executor_->CreateValue(value_pb));
  v0::Value output_pb;
  EXPECT_THAT(test_executor_->Materialize(id, &output_pb), IsOk());
  // Compare GraphDef protos without ordering.
  tensorflow::GraphDef input_graph_def;
  ASSERT_TRUE(input_graph_def.ParseFromString(graph_def));
  tensorflow::GraphDef materialized_graph_def;
  ASSERT_TRUE(materialized_graph_def.ParseFromString(
      output_pb.sequence().serialized_graph_def()));
  EXPECT_THAT(materialized_graph_def,
              testing::proto::IgnoringRepeatedFieldOrdering(
                  EqualsProto(input_graph_def)));
}

TEST_F(TensorFlowExecutorTest, CreateStructOneElement) {
  v0::Value input = TensorV(5);
  TFF_ASSERT_OK_AND_ASSIGN(auto value, test_executor_->CreateValue(input));
  TFF_ASSERT_OK_AND_ASSIGN(auto struct_, test_executor_->CreateStruct({value}));
  TFF_ASSERT_OK_AND_ASSIGN(auto output, test_executor_->Materialize(struct_));
  EXPECT_THAT(output, EqualsProto(StructV({input})));
}

TEST_F(TensorFlowExecutorTest, CreateStructSeveralElements) {
  v0::Value t1 = TensorV(5);
  v0::Value t2 = TensorV(6);
  v0::Value t3 = TensorV(7);
  v0::Value struct_ = StructV({TensorV(5), TensorV(6), TensorV(7)});
  TFF_ASSERT_OK_AND_ASSIGN(auto t1id, test_executor_->CreateValue(t1));
  TFF_ASSERT_OK_AND_ASSIGN(auto t2id, test_executor_->CreateValue(t2));
  TFF_ASSERT_OK_AND_ASSIGN(auto t3id, test_executor_->CreateValue(t3));
  TFF_ASSERT_OK_AND_ASSIGN(auto structid,
                           test_executor_->CreateStruct({t1id, t2id, t3id}));
  TFF_ASSERT_OK_AND_ASSIGN(auto output, test_executor_->Materialize(structid));
  EXPECT_THAT(output, EqualsProto(struct_));
}

TEST_F(TensorFlowExecutorTest, CreateSelectionFromCreateValue) {
  v0::Value input = StructV({TensorV(1), TensorV(2)});
  TFF_ASSERT_OK_AND_ASSIGN(auto vid, test_executor_->CreateValue(input));
  TFF_ASSERT_OK_AND_ASSIGN(auto t1id, test_executor_->CreateSelection(vid, 0));
  TFF_ASSERT_OK_AND_ASSIGN(auto t1, test_executor_->Materialize(t1id));
  EXPECT_THAT(t1, EqualsProto(TensorV(1)));
}

TEST_F(TensorFlowExecutorTest, CreateSelectionFromCreateStruct) {
  TFF_ASSERT_OK_AND_ASSIGN(auto t1id, test_executor_->CreateValue(TensorV(1)));
  TFF_ASSERT_OK_AND_ASSIGN(auto t2id, test_executor_->CreateValue(TensorV(2)));
  TFF_ASSERT_OK_AND_ASSIGN(auto structid,
                           test_executor_->CreateStruct({t1id, t2id}));
  TFF_ASSERT_OK_AND_ASSIGN(auto selectedid,
                           test_executor_->CreateSelection(structid, 1));
  TFF_ASSERT_OK_AND_ASSIGN(auto selected,
                           test_executor_->Materialize(selectedid));
  EXPECT_THAT(selected, EqualsProto(TensorV(2)));
}

TEST_F(TensorFlowExecutorTest, CreateSelectionNonStructImmediate) {
  TFF_ASSERT_OK_AND_ASSIGN(auto id, test_executor_->CreateValue(TensorV(1)));
  EXPECT_THAT(test_executor_->CreateSelection(id, 0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Cannot create selection on non-struct value."));
}

TEST_F(TensorFlowExecutorTest, CreateSelectionOOBImmediate) {
  TFF_ASSERT_OK_AND_ASSIGN(auto id, test_executor_->CreateValue(StructV({})));
  EXPECT_THAT(test_executor_->CreateSelection(id, 0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Attempted to access index 0 of a 0-length struct."));
}

TEST_F(TensorFlowExecutorTest, CallNoOutputTensors) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::ops::Placeholder x(root, tensorflow::DT_FLOAT);
  tensorflow::ops::Placeholder y(root, tensorflow::DT_FLOAT);
  // Ignore "sum".
  tensorflow::ops::AddV2 sum(root, x, y);

  auto in_binding = StructB({TensorB(x), TensorB(y)});

  // Check that we can make the output binding any old weird shape
  // so long as it has no tensors.
  auto out_binding = StructB({StructB({}), StructB({StructB({})})});

  v0::Value fn = ComputationV(in_binding, out_binding, root);
  v0::Value arg = StructV({TensorV(1.0), TensorV(2.0)});
  v0::Value expected = StructV({StructV({}), StructV({StructV({})})});
  CheckCallEqualsProto(fn, arg, expected);
  CheckCallRepeatedlyEqualsProto(fn, arg, expected);
}

TEST_F(TensorFlowExecutorTest, CallNoArgOneOutWithInitialize) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::TensorShape shape({3});
  tensorflow::ops::VarHandleOp var(root, tensorflow::DT_INT32, shape);
  auto var_init = tensorflow::ops::AssignVariableOp(
      root, var, tensorflow::ops::Const(root, {1, 2, 3}, shape));
  tensorflow::ops::ReadVariableOp read_var(root, var, tensorflow::DT_INT32);
  v0::Value fn = ComputationV(
      /*in_binding=*/absl::nullopt,
      /*out_binding=*/TensorB(read_var), root,
      /*init_op=*/var_init);
  tensorflow::Tensor expected(tensorflow::DT_INT32, shape);
  auto data = expected.flat<int32_t>();
  data(0) = 1;
  data(1) = 2;
  data(2) = 3;
  CheckCallEqualsProto(fn, absl::nullopt, TensorV(expected));
  // Ensure that repeatedly using the same session from the session provider
  // works correctly.
  CheckCallRepeatedlyEqualsProto(fn, absl::nullopt, TensorV(expected));
}

TEST_F(TensorFlowExecutorTest, CallOneInOut) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::ops::Placeholder in(root, tensorflow::DT_DOUBLE);
  tensorflow::ops::Identity out(root, in);
  v0::Value fn = ComputationV(TensorB(in), TensorB(out), root);
  v0::Value arg = TensorV(5.0);
  v0::Value expected = TensorV(5.0);
  CheckCallEqualsProto(fn, arg, expected);
}

TEST_F(TensorFlowExecutorTest, CallStructSwapInOut) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::ops::Placeholder xin(root, tensorflow::DT_DOUBLE);
  tensorflow::ops::Placeholder yin(root, tensorflow::DT_INT32);
  tensorflow::ops::Identity xout(root, xin);
  tensorflow::ops::Identity yout(root, yin);
  v0::Value fn = ComputationV(StructB({TensorB(xin), TensorB(yin)}),
                              StructB({TensorB(yout), TensorB(xout)}), root);
  v0::Value arg = StructV({TensorV(5.0), TensorV(1)});
  v0::Value expected = StructV({TensorV(1), TensorV(5.0)});
  CheckCallEqualsProto(fn, arg, expected);
}

TEST_F(TensorFlowExecutorTest, CallNestedStructSwapInOut) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::ops::Placeholder xin(root, tensorflow::DT_DOUBLE);
  tensorflow::ops::Placeholder yin(root, tensorflow::DT_INT32);
  tensorflow::ops::Identity xout(root, xin);
  tensorflow::ops::Identity yout(root, yin);
  v0::Value fn =
      ComputationV(StructB({StructB({TensorB(xin)}), TensorB(yin)}),
                   StructB({TensorB(yout), StructB({TensorB(xout)})}), root);
  v0::Value arg = StructV({StructV({TensorV(2.0)}), TensorV(4)});
  v0::Value expected = StructV({TensorV(4), StructV({TensorV(2.0)})});
  CheckCallEqualsProto(fn, arg, expected);
}

TEST_F(TensorFlowExecutorTest, CallAdd) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::ops::Placeholder x(root, tensorflow::DT_INT32);
  tensorflow::ops::Placeholder y(root, tensorflow::DT_INT32);
  tensorflow::ops::AddV2 out(root, x, y);
  v0::Value fn =
      ComputationV(StructB({TensorB(x), TensorB(y)}), TensorB(out), root);
  v0::Value arg = StructV({TensorV(1), TensorV(2)});
  v0::Value expected = TensorV(3);
  CheckCallEqualsProto(fn, arg, expected);
}

TEST_F(TensorFlowExecutorTest, StatefulCallGetsReinitialized) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::TensorShape shape({});
  tensorflow::ops::VarHandleOp var(root, tensorflow::DT_INT32, shape);
  tensorflow::ops::AssignVariableOp var_init(
      root, var, tensorflow::ops::Const(root, {0}, shape));
  tensorflow::ops::AssignAddVariableOp var_add_assign(
      root, var, tensorflow::ops::Const(root, {1}, shape));
  tensorflow::ops::ReadVariableOp read_var(
      root.WithControlDependencies({var_add_assign}), var,
      tensorflow::DT_INT32);
  v0::Value fn = ComputationV(absl::nullopt, TensorB(read_var), root, var_init);
  v0::Value expected = TensorV(1);
  CheckCallEqualsProto(fn, absl::nullopt, expected);
  CheckCallRepeatedlyEqualsProto(fn, absl::nullopt, expected);
  CheckCallParallelEqualsProto(fn, absl::nullopt, expected);
}

TEST_F(TensorFlowExecutorTest, CallWithComputationId) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::ops::Placeholder x(root, tensorflow::DT_INT32);
  tensorflow::ops::Placeholder y(root, tensorflow::DT_INT32);
  tensorflow::ops::AddV2 out(root, x, y);
  v0::Value fn =
      ComputationV(StructB({TensorB(x), TensorB(y)}), TensorB(out), root);
  // Add an ID to the value.
  fn.mutable_computation()->mutable_tensorflow()->mutable_cache_key()->set_id(
      1);
  v0::Value arg = StructV({TensorV(1), TensorV(2)});
  v0::Value expected = TensorV(3);
  // First call will populate the cache.
  CheckCallEqualsProto(fn, arg, expected);
  // Call a second time to exercise the cache.
  CheckCallEqualsProto(fn, arg, expected);
}

}  // namespace
}  // namespace tensorflow_federated

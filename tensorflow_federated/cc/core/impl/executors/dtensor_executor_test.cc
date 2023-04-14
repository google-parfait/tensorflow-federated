/* Copyright 2023, The TensorFlow Federated Authors.

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
#include "tensorflow_federated/cc/core/impl/executors/dtensor_executor.h"

#include <future>  //  NOLINT
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/proto/layout.pb.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow_federated/cc/core/impl/executors/dtensor_api.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"
#include "tensorflow_federated/cc/core/impl/executors/value_test_utils.h"
namespace tensorflow_federated {
namespace {
using ::tensorflow_federated::testing::EqualsProto;
using ::tensorflow_federated::testing::StructV;
using ::tensorflow_federated::testing::TensorV;
using ::tensorflow_federated::testing::TensorVFromIntList;

const char MESH_DIM_X[] = "x";

template <class TfOp>
inline v0::TensorFlow::Binding TensorB(const TfOp& op) {
  const tensorflow::Node* node = op.node();
  v0::TensorFlow::Binding binding;
  *binding.mutable_tensor()->mutable_tensor_name() = node->name();
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
    std::optional<v0::TensorFlow::Binding> in_binding,
    v0::TensorFlow::Binding out_binding, const tensorflow::Scope& scope,
    const std::optional<const tensorflow::Operation>& init_op = std::nullopt) {
  v0::Value value_pb;
  v0::Computation* comp_pb = value_pb.mutable_computation();
  // NOTE: we do not fill in the `type` field of `comp` because it is not needed
  // by the C++ TensorFlow executor.
  v0::TensorFlow* tensorflow_pb = comp_pb->mutable_tensorflow();
  tensorflow::GraphDef graphdef_pb;
  tensorflow::Status status = scope.ToGraphDef(&graphdef_pb);
  CHECK(status.ok()) << status;
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

tensorflow::dtensor::MeshProto CreateMeshForTest() {
  tensorflow::dtensor::MeshProto mesh;

  tensorflow::dtensor::MeshDimensionProto* dimension =
      mesh.add_mesh_dimensions();
  dimension->set_name(MESH_DIM_X);
  dimension->set_size(2);
  mesh.add_local_devices("/job:localhost/replica:0/task:0/device:CPU:0");
  mesh.add_local_devices("/job:localhost/replica:0/task:0/device:CPU:1");
  mesh.add_global_devices("/job:localhost/replica:0/task:0/device:CPU:0");
  mesh.add_global_devices("/job:localhost/replica:0/task:0/device:CPU:0");
  mesh.add_local_device_ids(0);
  mesh.add_local_device_ids(1);
  mesh.add_global_device_ids(0);
  mesh.add_global_device_ids(1);
  return mesh;
}

class DTensorConverterTestImpl : public DTensorConverter {
 public:
  ~DTensorConverterTestImpl() override = default;

  TFE_TensorHandle* TensorToDTensor(TFE_Context* context,
                                    TFE_TensorHandle* tensor_handle,
                                    const tensorflow::TF_Layout* layout,
                                    const char* device_name,
                                    TF_Status* status) override {
    auto* dtensor_input = TFE_DTENSOR_TensorToDTensor(
        context, tensor_handle, layout, device_name, status);
    tensorflow::ImmediateExecutionTensorHandle* dtensor =
        tensorflow::unwrap(dtensor_input);

    input_dtensors_.push_back(dtensor->DebugString());
    return dtensor_input;
  }

  // Wrapper for DTensor to Tensor conversion
  TFE_TensorHandle* DTensorToTensor(TFE_Context* context,
                                    TFE_TensorHandle* tensor_handle,
                                    const char* device_name,
                                    TF_Status* status) override {
    tensorflow::ImmediateExecutionTensorHandle* dtensor =
        tensorflow::unwrap(tensor_handle);
    result_dtensors_.push_back(dtensor->DebugString());
    return TFE_DTENSOR_DTensorToTensor(context, tensor_handle, device_name,
                                       status);
  };
  std::vector<std::string> input_dtensors_;
  std::vector<std::string> result_dtensors_;
};

class DTensorExecutorTest : public ::testing::Test {
 public:
  DTensorExecutorTest() {
    TF_Status* status = TF_NewStatus();
    std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)>
        opts(TFE_NewContextOptions(), TFE_DeleteContextOptions);
    context_ = TFE_NewContext(opts.get(), status);

    TFE_SetLogicalCpuDevices(context_, 2, "/job:localhost/replica:0/task:0",
                             status);
    mesh_ = CreateMeshForTest();
    device_name_ = "/job:localhost/replica:0/task:0/device:CUSTOM:1";
    auto mesh_or =
        tensorflow::dtensor::Mesh::ParseFromProto(CreateMeshForTest());
    CHECK(mesh_or.status().ok());

    auto mesh = mesh_or.value();
    TFE_DTENSOR_RegisterDTensorDevice(context_, tensorflow::wrap(&mesh),
                                      device_name_.c_str(), status);
    auto dtensor_converter = std::make_unique<DTensorConverterTestImpl>();
    // Keep a copy of pointer for testing before transferring ownership to
    // executor.
    dtensor_converter_ = dtensor_converter.get();
    test_executor_ = CreateDTensorExecutor(context_, device_name_, mesh,
                                           std::move(dtensor_converter), 10);

    TF_DeleteStatus(status);
  }

  ~DTensorExecutorTest() override { TFE_DeleteContext(context_); }

  TFE_Context* context_;
  tensorflow::dtensor::MeshProto mesh_;
  std::string device_name_;
  std::shared_ptr<Executor> test_executor_;
  DTensorConverterTestImpl* dtensor_converter_;
  void CheckRoundTrip(v0::Value& input_pb) {
    TFF_ASSERT_OK_AND_ASSIGN(OwnedValueId id,
                             test_executor_->CreateValue(input_pb));
    v0::Value output_pb;
    EXPECT_THAT(test_executor_->Materialize(id, &output_pb), IsOk());
    EXPECT_THAT(output_pb, testing::EqualsProto(input_pb));
  }

  void CheckCallEqualsProto(const v0::Value& fn,
                            const std::optional<v0::Value>& arg,
                            const v0::Value& expected) {
    TFF_ASSERT_OK_AND_ASSIGN(auto fn_id, test_executor_->CreateValue(fn));
    std::optional<OwnedValueId> arg_id;
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
                                      const std::optional<v0::Value>& arg,
                                      const v0::Value& expected) {
    TFF_ASSERT_OK_AND_ASSIGN(auto fn_id, test_executor_->CreateValue(fn));
    std::optional<OwnedValueId> arg_id;
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
};

TEST_F(DTensorExecutorTest, CallAddShardedLayout) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::ops::Placeholder x(root.WithOpName("input_x"),
                                 tensorflow::DT_INT32);
  tensorflow::ops::Placeholder y(root.WithOpName("input_y"),
                                 tensorflow::DT_INT32);
  tensorflow::ops::AddV2 out(root, x, y);
  v0::Value fn =
      ComputationV(StructB({TensorB(x), TensorB(y)}), TensorB(out), root);

  // Add a layout mapping for "input_x" argument to be sharded along X dimension
  // in the mesh.
  (*fn.mutable_computation()
        ->mutable_tensorflow()
        ->mutable_layout_map()
        ->mutable_name_to_sharding_spec())["input_x"] = MESH_DIM_X;
  v0::Value arg = StructV({TensorVFromIntList({1, 2, 3, 5}), TensorV(2)});
  v0::Value expected = TensorVFromIntList({3, 4, 5, 7});
  CheckCallEqualsProto(fn, arg, expected);

  ASSERT_EQ(dtensor_converter_->input_dtensors_.size(), 2);
  ASSERT_EQ(dtensor_converter_->result_dtensors_.size(), 1);

  // Input 0 is x sharded.
  EXPECT_THAT(dtensor_converter_->input_dtensors_[0],
              ::testing::HasSubstr("TensorHandle({\"CPU:0\": [1 2], \"CPU:1\": "
                                   "[3 5]}, layout=\"sharding_specs:x"));
  // Input 1 is scalar. No sharding.
  EXPECT_THAT(
      dtensor_converter_->input_dtensors_[1],
      ::testing::HasSubstr(
          "TensorHandle(2, layout=\"sharding_specs: "
          "mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0"));

  EXPECT_THAT(
      dtensor_converter_->result_dtensors_[0],
      AllOf(::testing::HasSubstr("dtype=DT_INT32"),
            ::testing::HasSubstr("{\"CPU:0\": [3 4], \"CPU:1\": [5 7]}"),
            ::testing::HasSubstr("sharding_specs:x")));
}

TEST_F(DTensorExecutorTest, CallAddReplicatedLayout) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::ops::Placeholder x(root.WithOpName("input_x"),
                                 tensorflow::DT_INT32);
  tensorflow::ops::Placeholder y(root.WithOpName("input_y"),
                                 tensorflow::DT_INT32);
  tensorflow::ops::AddV2 out(root, x, y);
  v0::Value fn =
      ComputationV(StructB({TensorB(x), TensorB(y)}), TensorB(out), root);

  // Add a layout mapping for "input_x" argument to be replicated in the mesh.
  (*fn.mutable_computation()
        ->mutable_tensorflow()
        ->mutable_layout_map()
        ->mutable_name_to_sharding_spec())["input_x"] = "unsharded";
  v0::Value arg = StructV({TensorVFromIntList({1, 2, 3, 5}), TensorV(2)});
  v0::Value expected = TensorVFromIntList({3, 4, 5, 7});
  CheckCallEqualsProto(fn, arg, expected);

  ASSERT_EQ(dtensor_converter_->input_dtensors_.size(), 2);
  ASSERT_EQ(dtensor_converter_->result_dtensors_.size(), 1);

  // Input 0 is unsharded.
  EXPECT_THAT(dtensor_converter_->input_dtensors_[0],
              ::testing::HasSubstr(
                  "TensorHandle([1 2 3 5], layout=\"sharding_specs:unsharded"));
  // Input 1 is scalar. No sharding.
  EXPECT_THAT(
      dtensor_converter_->input_dtensors_[1],
      ::testing::HasSubstr(
          "TensorHandle(2, layout=\"sharding_specs: "
          "mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0"));

  EXPECT_THAT(dtensor_converter_->result_dtensors_[0],
              AllOf(::testing::HasSubstr("dtype=DT_INT32"),
                    ::testing::HasSubstr("TensorHandle([3 4 5 7]"),
                    ::testing::HasSubstr("sharding_specs:unsharded")));
}

TEST_F(DTensorExecutorTest, CallAddNoLayoutSpecified) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::ops::Placeholder x(root.WithOpName("input_x"),
                                 tensorflow::DT_INT32);
  tensorflow::ops::Placeholder y(root.WithOpName("input_y"),
                                 tensorflow::DT_INT32);
  tensorflow::ops::AddV2 out(root, x, y);
  v0::Value fn =
      ComputationV(StructB({TensorB(x), TensorB(y)}), TensorB(out), root);

  v0::Value arg = StructV({TensorVFromIntList({1, 2, 3, 5}), TensorV(2)});
  v0::Value expected = TensorVFromIntList({3, 4, 5, 7});
  CheckCallEqualsProto(fn, arg, expected);

  // Confirm automatically unsharded layout is applied.
  ASSERT_EQ(dtensor_converter_->input_dtensors_.size(), 2);
  ASSERT_EQ(dtensor_converter_->result_dtensors_.size(), 1);

  // Input 0 is unsharded, if no specific sharding was specified.
  EXPECT_THAT(dtensor_converter_->input_dtensors_[0],
              ::testing::HasSubstr(
                  "TensorHandle([1 2 3 5], layout=\"sharding_specs:unsharded"));
  // Input 1 is scalar. No sharding.
  EXPECT_THAT(
      dtensor_converter_->input_dtensors_[1],
      ::testing::HasSubstr(
          "TensorHandle(2, layout=\"sharding_specs: "
          "mesh:|x=2|0,1|0,1|/job:localhost/replica:0/task:0/device:CPU:0"));

  EXPECT_THAT(dtensor_converter_->result_dtensors_[0],
              AllOf(::testing::HasSubstr("dtype=DT_INT32"),
                    ::testing::HasSubstr("TensorHandle([3 4 5 7]"),
                    ::testing::HasSubstr("sharding_specs:unsharded")));
}

TEST_F(DTensorExecutorTest, CallNoArgOneOutWithInitialize) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::TensorShape shape({4});
  tensorflow::ops::VarHandleOp var(root, tensorflow::DT_INT32, shape);
  auto var_init = tensorflow::ops::AssignVariableOp(
      root, var, tensorflow::ops::Const(root, {1, 2, 3, 4}, shape));
  tensorflow::ops::ReadVariableOp read_var(root, var, tensorflow::DT_INT32);
  v0::Value fn = ComputationV(
      /*in_binding=*/std::nullopt,
      /*out_binding=*/TensorB(read_var), root,
      /*init_op=*/var_init);
  (*fn.mutable_computation()
        ->mutable_tensorflow()
        ->mutable_layout_map()
        ->mutable_name_to_sharding_spec())["VarHandleOp"] = MESH_DIM_X;
  v0::Value expected = TensorVFromIntList({1, 2, 3, 4});
  this->CheckCallEqualsProto(fn, std::nullopt, expected);
  EXPECT_THAT(
      dtensor_converter_->result_dtensors_[0],
      AllOf(::testing::HasSubstr("dtype=DT_INT32"),
            ::testing::HasSubstr("{\"CPU:0\": [1 2], \"CPU:1\": [3 4]}"),
            ::testing::HasSubstr("sharding_specs:x")));
  // Ensure that repeatedly using the same session from the session provider
  // works correctly.
  this->CheckCallRepeatedlyEqualsProto(fn, std::nullopt, expected);
}
}  // namespace
}  // namespace tensorflow_federated

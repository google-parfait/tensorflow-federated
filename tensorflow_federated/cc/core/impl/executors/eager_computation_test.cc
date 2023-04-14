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
#include "tensorflow_federated/cc/core/impl/executors/eager_computation.h"

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "googlemock/include/gmock/gmock.h"
#include "googletest/include/gtest/gtest.h"
#include "absl/status/status.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/tf_datatype.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/no_op.h"
#include "tensorflow/cc/ops/resource_variable_ops.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/dtensor/cc/mesh_type.h"
#include "tensorflow_federated/cc/core/impl/executors/dtensor_api.h"
#include "tensorflow_federated/cc/core/impl/executors/status_matchers.h"

namespace tensorflow_federated {
namespace {

// TODO(b/256948367): Move these common methods to a base test utility file.
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

inline v0::Computation ComputationV(
    const tensorflow::Scope& scope,
    std::optional<v0::TensorFlow::Binding> in_binding,
    v0::TensorFlow::Binding out_binding,
    const std::optional<const tensorflow::Operation>& init_op = std::nullopt,
    const std::vector<tensorflow::FunctionDef> function_defs = {}) {
  v0::Computation comp_pb;
  v0::TensorFlow* tensorflow_pb = comp_pb.mutable_tensorflow();

  tensorflow::GraphDef graphdef_pb;

  tensorflow::Status status = scope.ToGraphDef(&graphdef_pb);
  CHECK(status.ok()) << status;

  if (!function_defs.empty()) {
    for (const auto& f : function_defs) {
      *graphdef_pb.mutable_library()->add_function() = f;
    }
  }

  tensorflow_pb->mutable_graph_def()->PackFrom(graphdef_pb);
  if (in_binding.has_value()) {
    *tensorflow_pb->mutable_parameter() = in_binding.value();
  }
  *tensorflow_pb->mutable_result() = out_binding;
  if (init_op.has_value()) {
    *tensorflow_pb->mutable_initialize_op() = init_op.value().node()->name();
  }
  return comp_pb;
}

inline v0::Computation ComputationV(tensorflow::FunctionDef func_def) {
  v0::Computation comp_pb;
  v0::TensorFlowFunction* tensorflow_pb = comp_pb.mutable_tensorflow_function();

  tensorflow::GraphDef graphdef_pb;

  tensorflow_pb->mutable_function_def()->PackFrom(func_def);
  return comp_pb;
}

TF_Tensor* FloatTensor(float v) {
  const int num_bytes = sizeof(float);
  float* values = new float[1];
  values[0] = v;
  return TF_NewTensor(
      TF_FLOAT, nullptr, 0, values, num_bytes,
      [](void* data, size_t, void*) { delete[] static_cast<float*>(data); },
      nullptr);
}

class EagerComputationTest : public ::testing::Test {};

TEST_F(EagerComputationTest, CallNoArgOneOutWithInitialize) {
  TF_Status* status = TF_NewStatus();
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status), TFE_DeleteContext);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::TensorShape shape({3});
  tensorflow::ops::VarHandleOp var(root, tensorflow::DT_INT32, shape);
  auto var_init = tensorflow::ops::AssignVariableOp(
      root, var, tensorflow::ops::Const(root, {1, 2, 3}, shape));
  tensorflow::ops::ReadVariableOp read_var(root, var, tensorflow::DT_INT32);

  auto fn = ComputationV(root, /*in_binding=*/std::nullopt,
                         /*out_binding=*/TensorB(read_var),
                         /*init_op=*/var_init);
  TFF_ASSERT_OK_AND_ASSIGN(auto comp,
                           EagerComputation::FromProto(fn.tensorflow()));
  TFF_ASSERT_OK_AND_ASSIGN(auto result, comp.Call(context.get(), std::nullopt));

  ASSERT_EQ(1, result.size());
  TFE_TensorHandle* result_tensor_handle = result[0];

  TF_Tensor* result_tensor =
      TFE_TensorHandleResolve(result_tensor_handle, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  int32_t expected[3];
  memcpy(&expected[0], TF_TensorData(result_tensor),
         TF_TensorByteSize(result_tensor));
  TF_DeleteTensor(result_tensor);
  TFE_DeleteTensorHandle(result_tensor_handle);
  EXPECT_EQ(1, expected[0]);
  EXPECT_EQ(2, expected[1]);
  EXPECT_EQ(3, expected[2]);

  TF_DeleteStatus(status);
}

TEST_F(EagerComputationTest, CallNoArgOneOutWithGroupedInitialize) {
  TF_Status* status = TF_NewStatus();
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status), TFE_DeleteContext);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::TensorShape shape({3});
  tensorflow::ops::VarHandleOp var(root, tensorflow::DT_INT32, shape);
  auto var_init = tensorflow::ops::AssignVariableOp(
      root, var, tensorflow::ops::Const(root, {1, 2, 3}, shape));
  tensorflow::ops::ReadVariableOp read_var(root, var, tensorflow::DT_INT32);

  auto grouped_initializer =
      tensorflow::ops::NoOp(root.WithOpName("grouped_initializer")
                                .WithControlDependencies({var_init}));
  auto fn = ComputationV(root, /*in_binding=*/std::nullopt,
                         /*out_binding=*/TensorB(read_var),
                         /*init_op=*/grouped_initializer);
  TFF_ASSERT_OK_AND_ASSIGN(auto comp,
                           EagerComputation::FromProto(fn.tensorflow()));
  TFF_ASSERT_OK_AND_ASSIGN(auto result, comp.Call(context.get(), std::nullopt));

  ASSERT_EQ(1, result.size());
  TFE_TensorHandle* result_tensor_handle = result[0];

  TF_Tensor* result_tensor =
      TFE_TensorHandleResolve(result_tensor_handle, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  int32_t expected[3];
  memcpy(&expected[0], TF_TensorData(result_tensor),
         TF_TensorByteSize(result_tensor));
  TF_DeleteTensor(result_tensor);
  TFE_DeleteTensorHandle(result_tensor_handle);
  EXPECT_EQ(1, expected[0]);
  EXPECT_EQ(2, expected[1]);
  EXPECT_EQ(3, expected[2]);

  TF_DeleteStatus(status);
}

TEST_F(EagerComputationTest, CallNoArgOneOutWithInitializeVariableWithLayout) {
  TF_Status* status = TF_NewStatus();
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status), TFE_DeleteContext);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::TensorShape shape({4});
  tensorflow::ops::VarHandleOp var(root, tensorflow::DT_INT32, shape);
  auto var_init = tensorflow::ops::AssignVariableOp(
      root, var, tensorflow::ops::Const(root, {1, 2, 3, 4}, shape));
  tensorflow::ops::ReadVariableOp read_var(root, var, tensorflow::DT_INT32);

  auto fn = ComputationV(root, /*in_binding=*/std::nullopt,
                         /*out_binding=*/TensorB(read_var),
                         /*init_op=*/var_init);

  TFE_SetLogicalCpuDevices(context.get(), 2, "/job:localhost/replica:0/task:0",
                           status);

  auto mesh = tensorflow::dtensor::Mesh::CreateMesh(
      "TestMesh",
      /*dim_names=*/{"x"},
      /*mesh_shape=*/{2},
      /*global_device_ids=*/{0, 1},
      /*global_devices_str=*/
      {"/job:localhost/task:0/device:CPU:0",
       "/job:localhost/task:0/device:CPU:1"},
      /*local_device_ids=*/{0, 1},
      /*local_devices_str=*/
      {"/job:localhost/task:0/device:CPU:0",
       "/job:localhost/task:0/device:CPU:1"},
      /*use_xla_spmd=*/false);

  std::string device_name = "/job:localhost/replica:0/task:0/device:CUSTOM:1";
  TFE_DTENSOR_RegisterDTensorDevice(context.get(), tensorflow::wrap(&mesh),
                                    device_name.c_str(), status);

  auto layout_or = tensorflow::dtensor::Layout::GetLayout({"x"}, mesh);
  ASSERT_TRUE(layout_or.ok());
  std::map<std::string, tensorflow::dtensor::Layout> layout_map{
      {"VarHandleOp", layout_or.value()}};

  TFF_ASSERT_OK_AND_ASSIGN(
      auto comp, EagerComputation::FromProto(fn.tensorflow(), layout_map));

  TFF_ASSERT_OK_AND_ASSIGN(auto result,
                           comp.Call(context.get(), std::nullopt, device_name));

  ASSERT_EQ(1, result.size());
  TFE_TensorHandle* result_dtensor_handle = result[0];
  tensorflow::ImmediateExecutionTensorHandle* dtensor =
      tensorflow::unwrap(result_dtensor_handle);
  std::string summary;
  ASSERT_TRUE(dtensor->SummarizeValue(summary).ok());
  EXPECT_THAT(summary, AllOf(testing::HasSubstr("CPU:0\": [1 2]"),
                             testing::HasSubstr("CPU:1\": [3 4]"),
                             testing::HasSubstr("sharding_specs:x")));

  auto result_tensor_handle = TFE_DTENSOR_DTensorToTensor(
      context.get(), result_dtensor_handle, device_name.c_str(), status);
  TF_Tensor* result_tensor =
      TFE_TensorHandleResolve(result_tensor_handle, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  int32_t expected[4];
  memcpy(&expected[0], TF_TensorData(result_tensor),
         TF_TensorByteSize(result_tensor));
  TF_DeleteTensor(result_tensor);
  TFE_DeleteTensorHandle(result_tensor_handle);
  TFE_DeleteTensorHandle(result_dtensor_handle);
  EXPECT_EQ(1, expected[0]);
  EXPECT_EQ(2, expected[1]);
  EXPECT_EQ(3, expected[2]);
  EXPECT_EQ(4, expected[3]);

  TF_DeleteStatus(status);
}

// Create a function def with the following flow:
// 1. Create Variable with VarHandleOp
// 2. Assign a const value to variable
// 3. Read the variable value
// 4. Return read variable value from function
tensorflow::FunctionDef VarFunctionDef() {
  tensorflow::FunctionDef def;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(
      R"pb(
        signature {
          name: "initialization_function"
          output_arg { name: "ReadVariableOp" type: DT_INT32 }
        }
        node_def {
          name: "VarHandleOp"
          op: "VarHandleOp"
          attr {
            key: "allowed_devices"
            value { list {} }
          }
          attr {
            key: "container"
            value { s: "" }
          }
          attr {
            key: "dtype"
            value { type: DT_INT32 }
          }
          attr {
            key: "shape"
            value { shape { dim { size: 4 } } }
          }
          attr {
            key: "shared_name"
            value { s: "" }
          }
        }
        node_def {
          name: "Const"
          op: "Const"
          attr {
            key: "dtype"
            value { type: DT_INT32 }
          }
          attr {
            key: "value"
            value {
              tensor {
                dtype: DT_INT32
                tensor_shape { dim { size: 4 } }
                tensor_content: "\001\000\000\000\002\000\000\000\003\000\000\000\004\000\000\000"
              }
            }
          }
        }
        node_def {
          name: "AssignVariableOp"
          op: "AssignVariableOp"
          input: "VarHandleOp:resource:0"
          input: "Const:output:0"
          attr {
            key: "dtype"
            value { type: DT_INT32 }
          }
          attr {
            key: "validate_shape"
            value { b: false }
          }
        }
        node_def {
          name: "ReadVariableOp_0"
          op: "ReadVariableOp"
          input: "VarHandleOp:resource:0"
          input: "^AssignVariableOp"
          attr {
            key: "dtype"
            value { type: DT_INT32 }
          }
        }
        ret { key: "ReadVariableOp" value: "ReadVariableOp_0:value:0" })pb",
      &def));
  return def;
}

// Create a function def with the following flow:
// 1. Create Variable with VarHandleOp
// 2. Read variable value.
// 3. Return read variable value from function (Should be empty since var is not
//    initialized.)
tensorflow::FunctionDef VarReadFunctionDef() {
  tensorflow::FunctionDef def;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(
      R"pb(
        signature {
          name: "initialization_function1"
          output_arg { name: "ReadVariableOp" type: DT_INT32 }
        }
        node_def {
          name: "VarHandleOp"
          op: "VarHandleOp"
          attr {
            key: "allowed_devices"
            value { list {} }
          }
          attr {
            key: "container"
            value { s: "" }
          }
          attr {
            key: "dtype"
            value { type: DT_INT32 }
          }
          attr {
            key: "shape"
            value { shape { dim { size: 3 } } }
          }
          attr {
            key: "shared_name"
            value { s: "" }
          }
        }
        node_def {
          name: "ReadVariableOp_0"
          op: "ReadVariableOp"
          input: "VarHandleOp:resource:0"
          attr {
            key: "dtype"
            value { type: DT_INT32 }
          }
        }
        ret { key: "ReadVariableOp" value: "ReadVariableOp_0:value:0" })pb",
      &def));
  return def;
}

TEST_F(EagerComputationTest, CallNoArgOneOutWithVariableCreationTest) {
  TF_Status* status = TF_NewStatus();
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status), TFE_DeleteContext);

  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  auto fn = ComputationV(VarFunctionDef());
  TFF_ASSERT_OK_AND_ASSIGN(
      auto comp, EagerComputation::FromProto(fn.tensorflow_function()));

  auto fn1 = ComputationV(VarReadFunctionDef());
  TFF_ASSERT_OK_AND_ASSIGN(
      auto comp1, EagerComputation::FromProto(fn1.tensorflow_function()));

  // Succeeds
  TFF_ASSERT_OK_AND_ASSIGN(auto result, comp.Call(context.get(), std::nullopt));

  ASSERT_EQ(1, result.size());
  TFE_TensorHandle* result_tensor_handle = result[0];

  TF_Tensor* result_tensor =
      TFE_TensorHandleResolve(result_tensor_handle, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  int32_t expected[4];
  memcpy(&expected[0], TF_TensorData(result_tensor),
         TF_TensorByteSize(result_tensor));
  TF_DeleteTensor(result_tensor);
  TFE_DeleteTensorHandle(result_tensor_handle);
  EXPECT_EQ(1, expected[0]);
  EXPECT_EQ(2, expected[1]);
  EXPECT_EQ(3, expected[2]);
  EXPECT_EQ(4, expected[3]);

  // With anonymous variable the read variable succeeds. But values should be
  // uninitialized. i.e. ensure that different vars are created. This is to
  // ensure that second call to the computation uses different variables.
  TFF_ASSERT_OK_AND_ASSIGN(auto result1,
                           comp1.Call(context.get(), std::nullopt));

  ASSERT_EQ(1, result1.size());
  TFE_TensorHandle* result1_tensor_handle = result1[0];

  TF_Tensor* result1_tensor =
      TFE_TensorHandleResolve(result1_tensor_handle, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  EXPECT_EQ(TF_TensorByteSize(result1_tensor), 0);

  TF_DeleteTensor(result1_tensor);
  TFE_DeleteTensorHandle(result1_tensor_handle);
  TF_DeleteStatus(status);
}

TEST_F(EagerComputationTest,
       CallNoArgOneOutWithVariableCreationWithLayoutTest) {
  TF_Status* status = TF_NewStatus();
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status), TFE_DeleteContext);

  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  auto fn = ComputationV(VarFunctionDef());

  TFE_SetLogicalCpuDevices(context.get(), 2, "/job:localhost/replica:0/task:0",
                           status);

  auto mesh = tensorflow::dtensor::Mesh::CreateMesh(
      "TestMesh",
      /*dim_names=*/{"x"},
      /*mesh_shape=*/{2},
      /*global_device_ids=*/{0, 1},
      /*global_devices_str=*/
      {"/job:localhost/task:0/device:CPU:0",
       "/job:localhost/task:0/device:CPU:1"},
      /*local_device_ids=*/{0, 1},
      /*local_devices_str=*/
      {"/job:localhost/task:0/device:CPU:0",
       "/job:localhost/task:0/device:CPU:1"},
      /*use_xla_spmd=*/false);

  std::string device_name = "/job:localhost/replica:0/task:0/device:CUSTOM:1";
  TFE_DTENSOR_RegisterDTensorDevice(context.get(), tensorflow::wrap(&mesh),
                                    device_name.c_str(), status);

  auto layout_or = tensorflow::dtensor::Layout::GetLayout({"x"}, mesh);
  ASSERT_TRUE(layout_or.ok());
  std::map<std::string, tensorflow::dtensor::Layout> layout_map{
      {"VarHandleOp", layout_or.value()}};

  TFF_ASSERT_OK_AND_ASSIGN(
      auto comp,
      EagerComputation::FromProto(fn.tensorflow_function(), layout_map));

  // Succeeds
  TFF_ASSERT_OK_AND_ASSIGN(auto result,
                           comp.Call(context.get(), std::nullopt, device_name));

  ASSERT_EQ(1, result.size());
  TFE_TensorHandle* result_dtensor_handle = result[0];

  tensorflow::ImmediateExecutionTensorHandle* dtensor =
      tensorflow::unwrap(result_dtensor_handle);
  std::string summary;
  ASSERT_TRUE(dtensor->SummarizeValue(summary).ok());
  EXPECT_THAT(summary, AllOf(testing::HasSubstr("CPU:0\": [1 2]"),
                             testing::HasSubstr("CPU:1\": [3 4]"),
                             testing::HasSubstr("sharding_specs:x")));

  auto result_tensor_handle = TFE_DTENSOR_DTensorToTensor(
      context.get(), result_dtensor_handle, device_name.c_str(), status);

  TF_Tensor* result_tensor =
      TFE_TensorHandleResolve(result_tensor_handle, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  int32_t expected[4];
  memcpy(&expected[0], TF_TensorData(result_tensor),
         TF_TensorByteSize(result_tensor));
  TF_DeleteTensor(result_tensor);
  TFE_DeleteTensorHandle(result_tensor_handle);
  TFE_DeleteTensorHandle(result_dtensor_handle);
  EXPECT_EQ(1, expected[0]);
  EXPECT_EQ(2, expected[1]);
  EXPECT_EQ(3, expected[2]);
  EXPECT_EQ(4, expected[3]);

  TF_DeleteStatus(status);
}

TEST_F(EagerComputationTest, CallAdd) {
  TF_Status* status = TF_NewStatus();
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status), TFE_DeleteContext);

  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::ops::Placeholder x(root, tensorflow::DT_FLOAT);
  tensorflow::ops::Placeholder y(root, tensorflow::DT_FLOAT);
  tensorflow::ops::AddV2 out(root, x, y);
  auto fn = ComputationV(root, StructB({TensorB(x), TensorB(y)}), TensorB(out));
  TFF_ASSERT_OK_AND_ASSIGN(auto comp,
                           EagerComputation::FromProto(fn.tensorflow()));

  auto* t_x = FloatTensor(5.);
  auto* t_y = FloatTensor(2.);
  TFE_TensorHandle* th_x = TFE_NewTensorHandle(t_x, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  TFE_TensorHandle* th_y = TFE_NewTensorHandle(t_x, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  std::vector<TFE_TensorHandle*> args;
  args.push_back(th_x);
  args.push_back(th_y);

  TFF_ASSERT_OK_AND_ASSIGN(auto result, comp.Call(context.get(), args));

  ASSERT_EQ(1, result.size());
  TFE_TensorHandle* result_tensor_handle = result[0];

  TF_Tensor* result_tensor =
      TFE_TensorHandleResolve(result_tensor_handle, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  EXPECT_EQ(10.0, *reinterpret_cast<float*>(TF_TensorData(result_tensor)));

  TF_DeleteTensor(result_tensor);
  TFE_DeleteTensorHandle(result_tensor_handle);

  TF_DeleteTensor(t_x);
  TFE_DeleteTensorHandle(th_x);

  TF_DeleteTensor(t_y);
  TFE_DeleteTensorHandle(th_y);

  TF_DeleteStatus(status);
}

TEST_F(EagerComputationTest, CallAddExtraPlaceholder) {
  TF_Status* status = TF_NewStatus();
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status), TFE_DeleteContext);

  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::ops::Placeholder x(root, tensorflow::DT_FLOAT);
  tensorflow::ops::Placeholder y(root, tensorflow::DT_FLOAT);
  // Not consumed placeholder. It should be ignored.
  tensorflow::ops::Placeholder z(root, tensorflow::DT_FLOAT);
  tensorflow::ops::AddV2 out(root, x, y);
  auto fn = ComputationV(root, StructB({TensorB(x), TensorB(y)}), TensorB(out));
  TFF_ASSERT_OK_AND_ASSIGN(auto comp,
                           EagerComputation::FromProto(fn.tensorflow()));

  auto* t_x = FloatTensor(5.);
  auto* t_y = FloatTensor(2.);

  TFE_TensorHandle* th_x = TFE_NewTensorHandle(t_x, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  TFE_TensorHandle* th_y = TFE_NewTensorHandle(t_x, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  std::vector<TFE_TensorHandle*> args;
  args.push_back(th_x);
  args.push_back(th_y);

  TFF_ASSERT_OK_AND_ASSIGN(auto result, comp.Call(context.get(), args));

  ASSERT_EQ(1, result.size());
  TFE_TensorHandle* result_tensor_handle = result[0];

  TF_Tensor* result_tensor =
      TFE_TensorHandleResolve(result_tensor_handle, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  EXPECT_EQ(10.0, *reinterpret_cast<float*>(TF_TensorData(result_tensor)));

  TF_DeleteTensor(result_tensor);
  TFE_DeleteTensorHandle(result_tensor_handle);

  TF_DeleteTensor(t_x);
  TFE_DeleteTensorHandle(th_x);

  TF_DeleteTensor(t_y);
  TFE_DeleteTensorHandle(th_y);

  TF_DeleteStatus(status);
}

tensorflow::FunctionDef AddFunctionDef() {
  tensorflow::FunctionDef def;
  CHECK(tensorflow::protobuf::TextFormat::ParseFromString(
      absl::StrCat("    signature {"
                   "      name: 'AddFunction'"
                   "      input_arg {"
                   "        name: 'a'"
                   "        type: DT_FLOAT"
                   "      }"
                   "      input_arg {"
                   "        name: 'b'"
                   "        type: DT_FLOAT"
                   "      }"
                   "      output_arg {"
                   "        name: 'sum'"
                   "        type: DT_FLOAT"
                   "      }"
                   "    }"
                   "    node_def {"
                   "      name: 'add'"
                   "      op: 'AddV2'"
                   "      input: 'a'"
                   "      input: 'b'",
                   "      attr {"
                   "        key: 'T'"
                   "        value {"
                   "          type: DT_FLOAT"
                   "        }"
                   "      }"
                   "    }"
                   "    ret {"
                   "      key: 'sum'"
                   "      value: 'add:z:0'"
                   "    }"),
      &def));
  return def;
}

TEST_F(EagerComputationTest, CallAddFunctionDef) {
  TF_Status* status = TF_NewStatus();
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status), TFE_DeleteContext);

  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  auto fn = ComputationV(AddFunctionDef());
  TFF_ASSERT_OK_AND_ASSIGN(
      auto comp, EagerComputation::FromProto(fn.tensorflow_function()));

  auto* t_x = FloatTensor(5.);
  auto* t_y = FloatTensor(2.);

  TFE_TensorHandle* th_x = TFE_NewTensorHandle(t_x, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  TFE_TensorHandle* th_y = TFE_NewTensorHandle(t_x, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  std::vector<TFE_TensorHandle*> args;
  args.push_back(th_x);
  args.push_back(th_y);

  TFF_ASSERT_OK_AND_ASSIGN(auto result, comp.Call(context.get(), args));

  ASSERT_EQ(1, result.size());
  TFE_TensorHandle* result_tensor_handle = result[0];

  TF_Tensor* result_tensor =
      TFE_TensorHandleResolve(result_tensor_handle, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  EXPECT_EQ(10.0, *reinterpret_cast<float*>(TF_TensorData(result_tensor)));

  TF_DeleteTensor(result_tensor);
  TFE_DeleteTensorHandle(result_tensor_handle);

  TF_DeleteTensor(t_x);
  TFE_DeleteTensorHandle(th_x);

  TF_DeleteTensor(t_y);
  TFE_DeleteTensorHandle(th_y);

  TF_DeleteStatus(status);
}

TEST_F(EagerComputationTest, CallAddGraphDefWithFunctionDef) {
  TF_Status* status = TF_NewStatus();
  std::unique_ptr<TFE_ContextOptions, decltype(&TFE_DeleteContextOptions)> opts(
      TFE_NewContextOptions(), TFE_DeleteContextOptions);
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context(
      TFE_NewContext(opts.get(), status), TFE_DeleteContext);

  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  auto add_function_def = AddFunctionDef();
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  tensorflow::ops::Placeholder x(root, tensorflow::DT_FLOAT);
  tensorflow::ops::Placeholder y(root, tensorflow::DT_FLOAT);

  tensorflow::OutputList placeholders;
  placeholders.push_back(x);
  placeholders.push_back(y);

  std::vector<tensorflow::DataType> tout_list;
  for (const auto& output : add_function_def.signature().output_arg()) {
    tout_list.push_back(output.type());
  }

  tensorflow::NameAttrList f_attr;
  f_attr.set_name(add_function_def.signature().name());

  tensorflow::ops::StatefulPartitionedCall call_op(root, placeholders,
                                                   tout_list, f_attr);
  tensorflow::ops::Identity identity(root, call_op.operation.output(0));
  auto fn = ComputationV(root, StructB({TensorB(x), TensorB(y)}),
                         TensorB(identity), std::nullopt, {add_function_def});
  TFF_ASSERT_OK_AND_ASSIGN(auto comp,
                           EagerComputation::FromProto(fn.tensorflow()));

  auto* t_x = FloatTensor(5.);
  auto* t_y = FloatTensor(2.);

  TFE_TensorHandle* th_x = TFE_NewTensorHandle(t_x, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  TFE_TensorHandle* th_y = TFE_NewTensorHandle(t_x, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  std::vector<TFE_TensorHandle*> args;
  args.push_back(th_x);
  args.push_back(th_y);

  TFF_ASSERT_OK_AND_ASSIGN(auto result, comp.Call(context.get(), args));

  ASSERT_EQ(1, result.size());
  TFE_TensorHandle* result_tensor_handle = result[0];

  TF_Tensor* result_tensor =
      TFE_TensorHandleResolve(result_tensor_handle, status);
  EXPECT_EQ(TF_GetCode(status), TF_OK) << TF_Message(status);

  EXPECT_EQ(10.0, *reinterpret_cast<float*>(TF_TensorData(result_tensor)));

  TF_DeleteTensor(result_tensor);
  TFE_DeleteTensorHandle(result_tensor_handle);

  TF_DeleteTensor(t_x);
  TFE_DeleteTensorHandle(th_x);

  TF_DeleteTensor(t_y);
  TFE_DeleteTensorHandle(th_y);

  TF_DeleteStatus(status);
}

TEST_F(EagerComputationTest, InvalidComputationProto) {
  v0::Computation comp_pb;
  v0::TensorFlow* tensorflow_pb = comp_pb.mutable_tensorflow();
  tensorflow::TensorProto tensor_pb;

  tensorflow_pb->mutable_graph_def()->PackFrom(tensor_pb);

  ASSERT_THAT(EagerComputation::FromProto(comp_pb.tensorflow()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}
}  // namespace

}  // namespace tensorflow_federated

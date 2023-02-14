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
#ifndef THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_EAGER_COMPUTATION_H_
#define THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_EAGER_COMPUTATION_H_

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"

namespace tensorflow_federated {
// Class responsible for converting a GraphDef into FunctionDef and executing it
// with TF2 eager C APIs.
// This class only owns the Function Defination derived from
// Computation->Tensorflow proto. It does not manage any resources or their
// lifetime.
//
// It provides a stateless interface `Call` for executing method with provided
// inputs. The callers of this class are expected to manage state.
class EagerComputation {
 public:
  // Converts GraphDef defined in Computation->Tensorflow into a FunctionDef
  // And instantiates EagerComputation class.
  // If non-empty Layout map is passed, a Relayout op is inserted after each
  // VarHandleOp node which has sharding spec specified.
  //
  // The placeholder inputs are expected to be sharded by the caller.
  // In DTensorExecutor, sharding for input bindings are applied before invoking
  // EagerComputation->Call.
  //
  // Note that Layout Map should only be specified when running EagerComputation
  // on a DTensor Mesh or with a DTensor device. "Relayout" op is only
  // recognized by DTensor device.
  static absl::StatusOr<EagerComputation> FromProto(
      const v0::TensorFlow& comp_pb,
      std::map<std::string, tensorflow::dtensor::Layout> layout_map = {});
  // Extracts FunctionDef defined in Computation->TensorFlowFunction and
  // instantiates EagerComputation class.
  // If non-empty Layout map is passed, a Relayout op is inserted after each
  // VarHandleOp node which has sharding spec specified.
  //
  // The placeholder inputs are expected to be sharded by the caller.
  // In DTensorExecutor, sharding for input bindings are applied before invoking
  // EagerComputation->Call.
  //
  // Note that Layout Map should only be specified when running EagerComputation
  // on a DTensor Mesh or with a DTensor device. "Relayout" op is only
  // recognized by DTensor device.
  static absl::StatusOr<EagerComputation> FromProto(
      const v0::TensorFlowFunction& comp_pb,
      std::map<std::string, tensorflow::dtensor::Layout> layout_map = {});

  EagerComputation(
      tensorflow::FunctionDef main_function_def,
      std::vector<tensorflow::FunctionDef> function_defs_to_register);

  // This method registers and executes TF `FunctionDef` owned by this object.
  //
  // Returns:
  //  - A vector of flattened output tensor handles on successful
  // execution. The results in this vector are in the iteration order of Result
  // binding in the Computation proto.
  //  - An error in case of failure to execute the function.
  //
  // This method is thread-safe i.e. it is safe to invoke `Call` with shared
  // context object.
  //
  // Invoking `Call` multiple times will execute the same FunctionDef
  // independently.
  //
  // Expected input to Call method is flattened list of input arguments, in the
  // iteration order of Parameter binding in `Computation` proto.
  // TODO(b/260640553): Allow register and call functions with same name but
  // different graphs in a thread safe manner.
  absl::StatusOr<std::vector<TFE_TensorHandle*>> Call(
      TFE_Context* context, std::optional<std::vector<TFE_TensorHandle*>> args,
      std::optional<std::string> device_name = std::nullopt);

 private:
  // Registers the FunctionDefs owned by the class object with TF eager context
  // provided in the input.
  absl::Status RegisterFunctions(TFE_Context* context);

  // Removes FunctionDef from context.
  absl::Status RemoveFunctions(TFE_Context* context);

  absl::Status ExecuteFunction(TFE_Context* context, std::string func_name,
                               std::optional<std::string> device_name,
                               absl::Span<TFE_TensorHandle*> args,
                               std::vector<TFE_TensorHandle*>* outputs);

  tensorflow::FunctionDef main_function_def_;
  std::vector<tensorflow::FunctionDef> function_defs_to_register_;
};
}  // namespace tensorflow_federated
#endif  // THIRD_PARTY_TENSORFLOW_FEDERATED_CC_CORE_IMPL_EXECUTORS_EAGER_COMPUTATION_H_

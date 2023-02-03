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
#include "tensorflow_federated/cc/core/impl/executors/dtensor_executor.h"

#include <cstdint>
#include <future>  // NOLINT
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow_federated/cc/core/impl/executors/dtensor_api.h"
#include "tensorflow_federated/cc/core/impl/executors/eager_computation.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/tensor_serialization.h"
#include "tensorflow_federated/cc/core/impl/executors/threading.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"
namespace tensorflow_federated {
namespace {

template <class T>
absl::StatusOr<T> ToAbslStatusOr(tensorflow::StatusOr<T> input) {
  if (input.ok()) {
    return input.value();
  }
  return tensorflow::ToAbslStatus(input.status());
}

class Value {
 public:
  // Method for materializing value as Value Proto.
  virtual absl::Status MaterializeValue(TFE_Context* context,
                                        v0::Value* value_pb,
                                        std::optional<std::string> device_name,
                                        ParallelTasks& tasks) = 0;

  // Method for executing computation with given arguments.
  // This method is relavent only for ComputationValue.
  virtual absl::StatusOr<std::shared_ptr<Value>> Call(
      std::optional<std::shared_ptr<Value>> arg, TFE_Context* context,
      std::optional<std::string> device_name) = 0;

  // Method to bind input parameters.
  // This is required to construct a flattened list of input arguments when
  // calling Computation.
  // TODO(b/256948367) If parameter name has a layout provided in layout map,
  // this method also converts input Tensor to DTensor with given layout.
  virtual absl::Status Bind(TFE_Context* context,
                            const v0::TensorFlow::Binding& shape,
                            std::vector<TFE_TensorHandle*>& bindings,
                            std::optional<std::string> device_name) = 0;

  // Returns value at given index.
  virtual absl::StatusOr<std::shared_ptr<Value>> ElementAt(int index) = 0;

  virtual ~Value() = default;
};

using ExecutorValue = std::shared_ptr<Value>;

absl::StatusOr<ExecutorValue> CreateValueAny(const v0::Value& value_pb);

class TensorValue : public Value {
 public:
  explicit TensorValue(TFE_TensorHandle* handle)
      : value_(std::unique_ptr<TFE_TensorHandle,
                               decltype(&TFE_DeleteTensorHandle)>(
            handle, TFE_DeleteTensorHandle)) {}

  static absl::StatusOr<ExecutorValue> CreateValue(const v0::Value& value_pb) {
    auto tensor = TFF_TRY(DeserializeTensorValue(value_pb));
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
        TF_NewStatus(), TF_DeleteStatus);
    return std::make_shared<TensorValue>(
        TFE_NewTensorHandle(tensor, status.get()));
  }

  absl::Status MaterializeValue(TFE_Context* context, v0::Value* value_pb,
                                std::optional<std::string> device_name,
                                ParallelTasks& tasks) override {
    return tasks.add_task([this, device_name, value_pb]() {
      std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
          TF_NewStatus(), TF_DeleteStatus);
      std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> tf_tensor(
          TFE_TensorHandleResolve(this->value_.get(), status.get()),
          TF_DeleteTensor);
      if (TF_GetCode(status.get()) != TF_OK) {
        return absl::InternalError(absl::StrCat("Tensor materialize failed: ",
                                                TF_Message(status.get())));
      }
      tensorflow::Tensor tensor;
      auto tf_status = tensorflow::TF_TensorToTensor(tf_tensor.get(), &tensor);
      if (!tf_status.ok()) {
        return absl::InternalError(absl::StrCat("Tensor materialize failed: ",
                                                tf_status.error_message()));
      }

      return SerializeTensorValue(tensor, value_pb);
    });
  }

  absl::StatusOr<ExecutorValue> Call(
      std::optional<ExecutorValue> arg, TFE_Context* context,
      std::optional<std::string> device_name) override {
    return absl::InvalidArgumentError(
        "Call method is allowed only for Computation");
  }

  absl::Status Bind(TFE_Context* context, const v0::TensorFlow::Binding& shape,
                    std::vector<TFE_TensorHandle*>& bindings,
                    std::optional<std::string> device_name) override {
    if (!shape.has_tensor()) {
      return absl::InvalidArgumentError(
          "Attempted to bind tensor value to non-tensor Binding.");
    }
    bindings.emplace_back(value_.get());
    return absl::OkStatus();
  }

  absl::StatusOr<std::shared_ptr<Value>> ElementAt(int index) override {
    return absl::InvalidArgumentError(
        "Cannot create selection on non-struct value.");
  }

 private:
  std::unique_ptr<TFE_TensorHandle, decltype(&TFE_DeleteTensorHandle)> value_;
};

class StructValue : public Value {
 public:
  static absl::StatusOr<ExecutorValue> CreateValue(const v0::Value& value_pb) {
    if (!value_pb.has_struct_()) {
      return absl::InvalidArgumentError(
          "Creating StructValue from a non-struct value proto.");
    }
    std::vector<ExecutorValue> values;
    values.reserve(value_pb.struct_().element_size());
    for (const auto& element : value_pb.struct_().element()) {
      auto value = TFF_TRY(CreateValueAny(element.value()));
      values.push_back(value);
    }
    return std::make_shared<StructValue>(values);
  }

  explicit StructValue(std::vector<ExecutorValue> values) : values_(values) {}

  absl::Status MaterializeValue(TFE_Context* context, v0::Value* value_pb,
                                std::optional<std::string> device_name,
                                ParallelTasks& tasks) override {
    v0::Value::Struct* struct_pb = value_pb->mutable_struct_();
    for (const ExecutorValue& element : values_) {
      TFF_TRY(element->MaterializeValue(
          context, struct_pb->add_element()->mutable_value(), device_name,
          tasks));
    }
    return absl::OkStatus();
  }

  absl::StatusOr<ExecutorValue> Call(
      std::optional<ExecutorValue> arg, TFE_Context* context,
      std::optional<std::string> device_name) override {
    return absl::InvalidArgumentError(
        "Call method is allowed only for Computation");
  }

  absl::StatusOr<ExecutorValue> ElementAt(int index) override {
    if (values_.size() <= index || index < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("Attempted to access index ", index, " of a ",
                       values_.size(), "-length struct."));
    }
    return ExecutorValue(values_[index]);
  }

  absl::Status Bind(TFE_Context* context, const v0::TensorFlow::Binding& shape,
                    std::vector<TFE_TensorHandle*>& bindings,
                    std::optional<std::string> device_name) override {
    if (!shape.has_struct_()) {
      return absl::InvalidArgumentError(
          "Attempted to bind struct value to non-struct Binding.");
    }
    if (shape.struct_().element_size() != values_.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Attempted to bind struct with ", values_.size(),
                       " fields to an argument struct with ",
                       shape.struct_().element_size(), " fields."));
    }
    for (int i = 0; i < values_.size(); i++) {
      TFF_TRY(values_[i]->Bind(context, shape.struct_().element(i), bindings,
                               device_name));
    }
    return absl::OkStatus();
  }

 private:
  std::vector<ExecutorValue> values_;
};

// Returns Executor Value in flattened order as per bindings. This follows the
// same semantics as used by Bind.
// TODO(b/256948367): Extract this function to a common library, since used by
// both tensorflow_executor and dtensor_executor.
absl::StatusOr<ExecutorValue> FromTensorsAndBindingStructure(
    const v0::TensorFlow::Binding& binding_structure,
    absl::Span<TFE_TensorHandle*>* tensors) {
  switch (binding_structure.binding_case()) {
    case v0::TensorFlow::Binding::kTensor: {
      if (tensors->empty()) {
        return absl::InternalError(
            "TensorFlow computation had fewer output tensors than expected.");
      }
      auto tensor_val = std::make_shared<TensorValue>(tensors->front());
      tensors->remove_prefix(1);
      return tensor_val;
    }
    case v0::TensorFlow::Binding::kStruct: {
      auto elements = std::vector<ExecutorValue>();
      elements.reserve(binding_structure.struct_().element_size());
      for (const auto& e_structure : binding_structure.struct_().element()) {
        elements.push_back(
            TFF_TRY(FromTensorsAndBindingStructure(e_structure, tensors)));
      }
      return std::make_shared<StructValue>(elements);
    }
    default: {
      return absl::UnimplementedError(absl::StrCat(
          "Unknown output binding kind: ", binding_structure.binding_case()));
    }
  }
}

class ComputationValue : public Value {
 public:
  static absl::StatusOr<ExecutorValue> CreateValue(const v0::Value& value_pb) {
    if (!value_pb.has_computation()) {
      return absl::InvalidArgumentError(
          "Creating ComputationValue from a non-computation value proto.");
    }
    auto eager_comp = TFF_TRY(
        EagerComputation::FromProto(value_pb.computation().tensorflow()));
    return std::make_shared<ComputationValue>(
        eager_comp,
        value_pb.computation().tensorflow().has_parameter()
            ? std::optional(value_pb.computation().tensorflow().parameter())
            : std::nullopt,
        value_pb.computation().tensorflow().result());
  }

  ComputationValue(EagerComputation computation,
                   std::optional<v0::TensorFlow::Binding> parameter_shape,
                   v0::TensorFlow::Binding output_shape)
      : computation_(computation),
        parameter_shape_(parameter_shape),
        output_shape_(output_shape) {}

  absl::Status MaterializeValue(TFE_Context* context, v0::Value* value_pb,
                                std::optional<std::string> device_name,
                                ParallelTasks& tasks) override {
    return absl::InvalidArgumentError(
        "Cannot materialize uncalled computations");
  }

  absl::StatusOr<ExecutorValue> Call(
      std::optional<ExecutorValue> arg, TFE_Context* context,
      std::optional<std::string> device_name) override {
    std::vector<TFE_TensorHandle*> flattened_inputs;
    if (arg.has_value()) {
      TFF_TRY(arg.value()->Bind(context, parameter_shape_.value(),
                                flattened_inputs, device_name));
    }
    auto outputs =
        TFF_TRY(computation_.Call(context, flattened_inputs, device_name));
    absl::Span<TFE_TensorHandle*> outputs_span(outputs);
    return FromTensorsAndBindingStructure(output_shape_, &outputs_span);
  }

  absl::Status Bind(TFE_Context* context, const v0::TensorFlow::Binding& shape,
                    std::vector<TFE_TensorHandle*>& bindings,
                    std::optional<std::string> device_name) override {
    return absl::InvalidArgumentError(
        "Attempted to bind computation value as argument to a TensorFlow "
        "computation. This is not supported.");
  }

  absl::StatusOr<std::shared_ptr<Value>> ElementAt(int index) override {
    return absl::InvalidArgumentError(
        "Cannot create selection on non-struct value.");
  }

 private:
  EagerComputation computation_;
  std::optional<v0::TensorFlow::Binding> parameter_shape_;
  v0::TensorFlow::Binding output_shape_;
};

absl::StatusOr<ExecutorValue> CreateValueAny(const v0::Value& value_pb) {
  VLOG(2) << "Creating value: " << value_pb.Utf8DebugString();
  switch (value_pb.value_case()) {
    case v0::Value::kTensor: {
      return TensorValue::CreateValue(value_pb);
    }
    case v0::Value::kStruct: {
      return StructValue::CreateValue(value_pb);
    }
    case v0::Value::kComputation: {
      return ComputationValue::CreateValue(value_pb);
    }
    case v0::Value::kSequence: {
      return absl::UnimplementedError("Sequence is not implemented yet.");
    }
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unknown value proto type ", value_pb.value_case()));
  }
}

using ValueFuture = std::shared_future<absl::StatusOr<ExecutorValue>>;

class DTensorExecutor : public ExecutorBase<ValueFuture> {
 public:
  DTensorExecutor(
      std::optional<std::string> dtensor_device_name,
      std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context,
      int32_t max_concurrent_computation_calls)
      : context_(std::move(context)),
        dtensor_device_name_(dtensor_device_name),
        max_concurrent_computation_calls_(max_concurrent_computation_calls),
        thread_pool_(
            // Use a threadpool with CPU * 4 or the user specified
            // maximum.
            ((max_concurrent_computation_calls > 0)
                 ? max_concurrent_computation_calls
                 : std::thread::hardware_concurrency() * 4),
            ExecutorName()) {
    VLOG(2) << "max_concurrent_computation_calls: "
            << max_concurrent_computation_calls_;
    VLOG(2) << "thread pool size: "
            << ((max_concurrent_computation_calls > 0)
                    ? max_concurrent_computation_calls
                    : std::thread::hardware_concurrency() * 4);
  }

 protected:
  absl::string_view ExecutorName() final { return "DTensorExecutor"; }
  absl::StatusOr<ValueFuture> CreateExecutorValue(
      const v0::Value& value_pb) final {
    VLOG(2) << "Creating value: " << value_pb.Utf8DebugString();
    return ThreadRun(
        [value_pb]() -> absl::StatusOr<ExecutorValue> {
          return CreateValueAny(value_pb);
        },
        &thread_pool_);
  }

  absl::StatusOr<ValueFuture> CreateCall(
      ValueFuture function, std::optional<ValueFuture> argument) final {
    return ThreadRun(
        [this, function = std::move(function),
         argument = std::move(argument)]() -> absl::StatusOr<ExecutorValue> {
          ExecutorValue fn = TFF_TRY(Wait(function));
          std::optional<ExecutorValue> arg = std::nullopt;
          if (argument.has_value()) {
            arg = TFF_TRY(Wait(argument.value()));
          }
          return fn->Call(arg, this->context_.get(),
                          this->dtensor_device_name_);
        },
        &thread_pool_);
  }

  absl::StatusOr<ValueFuture> CreateStruct(
      std::vector<ValueFuture> members) final {
    return Map(
        std::move(members),
        [](std::vector<ExecutorValue>&& elements)
            -> absl::StatusOr<ExecutorValue> {
          return (std::make_shared<StructValue>(std::move(elements)));
        },
        &thread_pool_);
  }
  absl::StatusOr<ValueFuture> CreateSelection(ValueFuture value,
                                              const uint32_t index) final {
    return Map(
        std::vector<ValueFuture>({value}),
        [index](std::vector<ExecutorValue>&& values)
            -> absl::StatusOr<ExecutorValue> {
          // Note that we could be attempting to select from a result of a
          // function call (which might be either a tensor or a structure, or a
          // nested structure, etc). So I think this implies that we will need
          // to implement Call to respect this invariant (if your function
          // returns a structure, you will need to create a StructValue)
          // TODO(b/256948367): Confirm that createSelection from result of
          // CreateCall works as expected.
          return values[0]->ElementAt(index);
        },
        &thread_pool_);
  }

  absl::Status Materialize(ValueFuture value_fut, v0::Value* value_pb) final {
    ExecutorValue value = TFF_TRY(Wait(std::move(value_fut)));
    ParallelTasks tasks(&thread_pool_);
    TFF_TRY(value->MaterializeValue(context_.get(), value_pb,
                                    dtensor_device_name_, tasks));
    TFF_TRY(tasks.WaitAll());
    return absl::OkStatus();
  }

 private:
  std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context_;
  std::optional<std::string> dtensor_device_name_;
  int32_t max_concurrent_computation_calls_;
  ThreadPool thread_pool_;
};

}  // namespace

std::shared_ptr<Executor> CreateDTensorExecutor(
    std::optional<std::string> dtensor_device_name,
    std::unique_ptr<TFE_Context, decltype(&TFE_DeleteContext)> context,
    int32_t max_concurrent_computation_calls) {
  return std::make_shared<DTensorExecutor>(dtensor_device_name,
                                           std::move(context),
                                           max_concurrent_computation_calls);
}
}  // namespace tensorflow_federated

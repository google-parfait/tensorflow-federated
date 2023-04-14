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
#include <string_view>
#include <thread>  // NOLINT
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/c/eager/c_api.h"
#include "tensorflow/c/eager/tfe_tensorhandle_internal.h"
#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_tensor.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow_federated/cc/core/impl/executors/dtensor_api.h"
#include "tensorflow_federated/cc/core/impl/executors/eager_computation.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/tensor_serialization.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_status_compat.h"
#include "tensorflow_federated/cc/core/impl/executors/tensorflow_utils.h"
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

class DTensorConverterImpl : public DTensorConverter {
 public:
  ~DTensorConverterImpl() override = default;

  TFE_TensorHandle* TensorToDTensor(TFE_Context* context,
                                    TFE_TensorHandle* tensor_handle,
                                    const tensorflow::TF_Layout* layout,
                                    const char* device_name,
                                    TF_Status* status) override {
    return TFE_DTENSOR_TensorToDTensor(context, tensor_handle, layout,
                                       device_name, status);
  }

  // Wrapper for DTensor to Tensor conversion
  TFE_TensorHandle* DTensorToTensor(TFE_Context* context,
                                    TFE_TensorHandle* tensor_handle,
                                    const char* device_name,
                                    TF_Status* status) override {
    return TFE_DTENSOR_DTensorToTensor(context, tensor_handle, device_name,
                                       status);
  };
};

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
  virtual absl::Status Bind(
      TFE_Context* context, const v0::TensorFlow::Binding& shape,
      const std::map<std::string, tensorflow::dtensor::Layout>& layout_map,
      std::vector<TFE_TensorHandle*>& bindings,
      std::optional<std::string> device_name,
      std::optional<const tensorflow::dtensor::Mesh> mesh) = 0;

  // Returns value at given index.
  virtual absl::StatusOr<std::shared_ptr<Value>> ElementAt(int index) = 0;

  virtual ~Value() = default;
};

using ExecutorValue = std::shared_ptr<Value>;

absl::StatusOr<ExecutorValue> CreateValueAny(
    const v0::Value& value_pb,
    std::optional<tensorflow::dtensor::Mesh> mesh = std::nullopt,
    DTensorConverter* converter = nullptr);

class TensorValue : public Value {
 public:
  explicit TensorValue(TFE_TensorHandle* handle, DTensorConverter* converter)
      : value_(std::unique_ptr<TFE_TensorHandle,
                               decltype(&TFE_DeleteTensorHandle)>(
            handle, TFE_DeleteTensorHandle)),
        converter_(converter) {}

  static absl::StatusOr<ExecutorValue> CreateTensor(
      const v0::Value& value_pb, DTensorConverter* converter) {
    auto tensor = TFF_TRY(DeserializeTensorValue(value_pb));
    std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
        TF_NewStatus(), TF_DeleteStatus);
    return std::make_shared<TensorValue>(
        TFE_NewTensorHandle(tensor, status.get()), converter);
  }

  TF_Tensor* GetTensorValue(TFE_Context* context,
                            std::optional<std::string> device_name,
                            TF_Status* status) {
    if (device_name.has_value()) {
      bool is_dtensor_value = TFE_DTENSOR_IsTensorHandleOnDevice(
          context, this->value_.get(), device_name.value().c_str(), status);
      if (TF_GetCode(status) != TF_OK) {
        return nullptr;
      }
      if (is_dtensor_value) {
        std::unique_ptr<TFE_TensorHandle, decltype(&TFE_DeleteTensorHandle)>
            tensor_handle_from_dtensor(converter_->DTensorToTensor(
                                           context, this->value_.get(),
                                           device_name.value().c_str(), status),
                                       TFE_DeleteTensorHandle);
        if (TF_GetCode(status) != TF_OK) {
          return nullptr;
        }
        return TFE_TensorHandleResolve(tensor_handle_from_dtensor.get(),
                                       status);
      }
    }
    return TFE_TensorHandleResolve(this->value_.get(), status);
  }

  absl::Status MaterializeValue(TFE_Context* context, v0::Value* value_pb,
                                std::optional<std::string> device_name,
                                ParallelTasks& tasks) override {
    return tasks.add_task([this, device_name, value_pb, context]() {
      std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
          TF_NewStatus(), TF_DeleteStatus);

      std::unique_ptr<TF_Tensor, decltype(&TF_DeleteTensor)> tf_tensor(
          GetTensorValue(context, device_name, status.get()), TF_DeleteTensor);
      if (TF_GetCode(status.get()) != TF_OK) {
        return absl::InternalError(absl::StrCat("Tensor materialize failed: ",
                                                TF_Message(status.get())));
      }
      tensorflow::Tensor tensor;
      auto tf_status = tensorflow::TF_TensorToTensor(tf_tensor.get(), &tensor);
      if (!tf_status.ok()) {
        return absl::InternalError(
            absl::StrCat("Tensor materialize failed: ", ToMessage(tf_status)));
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

  absl::Status Bind(
      TFE_Context* context, const v0::TensorFlow::Binding& shape,
      const std::map<std::string, tensorflow::dtensor::Layout>& layout_map,
      std::vector<TFE_TensorHandle*>& bindings,
      std::optional<std::string> device_name,
      std::optional<const tensorflow::dtensor::Mesh> mesh) override {
    if (!shape.has_tensor()) {
      return absl::InvalidArgumentError(
          "Attempted to bind tensor value to non-tensor Binding.");
    }
    if (mesh.has_value() && device_name.has_value()) {
      std::unique_ptr<TF_Status, decltype(&TF_DeleteStatus)> status(
          TF_NewStatus(), TF_DeleteStatus);
      // Binding names have suffix ":N" in them, remove that to look up nodes.
      auto it = layout_map.find(GetNodeName(shape.tensor().tensor_name()));
      tensorflow::dtensor::Layout layout;
      if (it != layout_map.end() && device_name.has_value()) {
        layout = it->second;
      } else {
        // If layout map does not have sharding specified for the Tensor,
        // place the tensor on device with replicated layout.
        auto value_rank = TFE_TensorHandleNumDims(value_.get(), status.get());
        layout = tensorflow::dtensor::Layout::ReplicatedOnMesh(mesh.value(),
                                                               value_rank);
      }
      // Create DTensor with layout and use it as arg to function.
      auto* dtensor_value = converter_->TensorToDTensor(
          context, value_.get(), tensorflow::wrap(&layout),
          device_name.value().c_str(), status.get());
      if (TF_GetCode(status.get()) != TF_OK) {
        return absl::InternalError(absl::StrCat("dtensor creation failed.. ",
                                                TF_Message(status.get())));
      }
      bindings.emplace_back(dtensor_value);
    } else {
      // If not running on a DTensor device, copy the tensor handle into binding
      // directly.
      bindings.emplace_back(value_.get());
    }
    return absl::OkStatus();
  }

  absl::StatusOr<std::shared_ptr<Value>> ElementAt(int index) override {
    return absl::InvalidArgumentError(
        "Cannot create selection on non-struct value.");
  }

 private:
  std::unique_ptr<TFE_TensorHandle, decltype(&TFE_DeleteTensorHandle)> value_;
  DTensorConverter* converter_ = nullptr;
};

class StructValue : public Value {
 public:
  static absl::StatusOr<ExecutorValue> CreateStruct(
      const v0::Value& value_pb, std::optional<tensorflow::dtensor::Mesh> mesh,
      DTensorConverter* converter) {
    if (!value_pb.has_struct_()) {
      return absl::InvalidArgumentError(
          "Creating StructValue from a non-struct value proto.");
    }
    std::vector<ExecutorValue> values;
    values.reserve(value_pb.struct_().element_size());
    for (const auto& element : value_pb.struct_().element()) {
      auto value = TFF_TRY(CreateValueAny(element.value(), mesh, converter));
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

  absl::Status Bind(
      TFE_Context* context, const v0::TensorFlow::Binding& shape,
      const std::map<std::string, tensorflow::dtensor::Layout>& layout_map,
      std::vector<TFE_TensorHandle*>& bindings,
      std::optional<std::string> device_name,
      std::optional<const tensorflow::dtensor::Mesh> mesh) override {
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
      TFF_TRY(values_[i]->Bind(context, shape.struct_().element(i), layout_map,
                               bindings, device_name, mesh));
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
    absl::Span<TFE_TensorHandle*>* tensors,
    DTensorConverter* converter = nullptr) {
  switch (binding_structure.binding_case()) {
    case v0::TensorFlow::Binding::kTensor: {
      if (tensors->empty()) {
        return absl::InternalError(
            "TensorFlow computation had fewer output tensors than expected.");
      }
      auto tensor_val =
          std::make_shared<TensorValue>(tensors->front(), converter);
      tensors->remove_prefix(1);
      return tensor_val;
    }
    case v0::TensorFlow::Binding::kStruct: {
      auto elements = std::vector<ExecutorValue>();
      elements.reserve(binding_structure.struct_().element_size());
      for (const auto& e_structure : binding_structure.struct_().element()) {
        elements.push_back(TFF_TRY(
            FromTensorsAndBindingStructure(e_structure, tensors, converter)));
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
  static absl::StatusOr<ExecutorValue> CreateComputation(
      const v0::Value& value_pb, std::optional<tensorflow::dtensor::Mesh> mesh,
      DTensorConverter* converter) {
    if (!value_pb.has_computation()) {
      return absl::InvalidArgumentError(
          "Creating ComputationValue from a non-computation value proto.");
    }
    if (value_pb.computation().has_tensorflow()) {
      std::map<std::string, tensorflow::dtensor::Layout> layout_map;
      // Use layout information only when Mesh is present
      if (mesh.has_value() &&
          value_pb.computation().tensorflow().has_layout_map()) {
        for (const auto& sharding : value_pb.computation()
                                        .tensorflow()
                                        .layout_map()
                                        .name_to_sharding_spec()) {
          std::vector<std::string> sharding_specs =
              absl::StrSplit(sharding.second, ',');
          auto layout_or = tensorflow::dtensor::Layout::GetLayout(
              sharding_specs, mesh.value());
          if (!layout_or.ok()) {
            return tensorflow::ToAbslStatus(layout_or.status());
          }
          layout_map[sharding.first] = layout_or.value();
        }
      }

      auto eager_comp = TFF_TRY(EagerComputation::FromProto(
          value_pb.computation().tensorflow(), layout_map));
      return std::make_shared<ComputationValue>(
          eager_comp,
          value_pb.computation().tensorflow().has_parameter()
              ? std::optional(value_pb.computation().tensorflow().parameter())
              : std::nullopt,
          value_pb.computation().tensorflow().result(), layout_map, mesh,
          converter);
    }
    // TODO(b/256948367): Add creating eager_computation object from
    // Computation->tensorflow_function.
    return absl::InvalidArgumentError(
        "Only tensorflow Computation type is supported.");
  }

  ComputationValue(
      EagerComputation computation,
      std::optional<v0::TensorFlow::Binding> parameter_shape,
      v0::TensorFlow::Binding output_shape,
      std::map<std::string, tensorflow::dtensor::Layout> layout_map,
      std::optional<const tensorflow::dtensor::Mesh> mesh,
      DTensorConverter* converter)
      : computation_(computation),
        parameter_shape_(parameter_shape),
        output_shape_(output_shape),
        layout_map_(std::move(layout_map)),
        mesh_(mesh),
        converter_(converter) {}

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
      TFF_TRY(arg.value()->Bind(context, parameter_shape_.value(), layout_map_,
                                flattened_inputs, device_name, mesh_));
    }
    auto outputs =
        TFF_TRY(computation_.Call(context, flattened_inputs, device_name));

    if (mesh_.has_value() && device_name.has_value()) {
      // When mesh and device are present, a DTensor handle corresponding to
      // input tensor value is created. These should be deleted after the
      // execution.
      //
      // Tensor handles inside TensorValue are deleted at the destruction.
      // However, DTensor handles are only created based on layouts specified in
      // the computation proto just before executing Computation->Call. These
      // should be cleaned up explicitly after the call is complete.
      for (TFE_TensorHandle* handle : flattened_inputs) {
        TFE_DeleteTensorHandle(handle);
      }
    }
    absl::Span<TFE_TensorHandle*> outputs_span(outputs);
    return FromTensorsAndBindingStructure(output_shape_, &outputs_span,
                                          converter_);
  }

  absl::Status Bind(
      TFE_Context* context, const v0::TensorFlow::Binding& shape,
      const std::map<std::string, tensorflow::dtensor::Layout>& layout_map,
      std::vector<TFE_TensorHandle*>& bindings,
      std::optional<std::string> device_name,
      std::optional<const tensorflow::dtensor::Mesh> mesh) override {
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
  std::map<std::string, tensorflow::dtensor::Layout> layout_map_;
  std::optional<const tensorflow::dtensor::Mesh> mesh_;
  DTensorConverter* converter_ = nullptr;
};

absl::StatusOr<ExecutorValue> CreateValueAny(
    const v0::Value& value_pb, std::optional<tensorflow::dtensor::Mesh> mesh,
    DTensorConverter* converter) {
  VLOG(2) << "Creating value: " << value_pb.Utf8DebugString();
  switch (value_pb.value_case()) {
    case v0::Value::kTensor: {
      return TensorValue::CreateTensor(value_pb, converter);
    }
    case v0::Value::kStruct: {
      return StructValue::CreateStruct(value_pb, mesh, converter);
    }
    case v0::Value::kComputation: {
      return ComputationValue::CreateComputation(value_pb, mesh, converter);
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
  DTensorExecutor(TFE_Context* context,
                  std::optional<std::string> dtensor_device_name,
                  std::optional<tensorflow::dtensor::Mesh> mesh,
                  std::unique_ptr<DTensorConverter> converter,
                  int32_t max_concurrent_computation_calls)
      : context_(std::move(context)),
        dtensor_device_name_(dtensor_device_name),
        max_concurrent_computation_calls_(max_concurrent_computation_calls),
        mesh_(mesh),
        converter_(std::move(converter)),
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
  std::string_view ExecutorName() final { return "DTensorExecutor"; }
  absl::StatusOr<ValueFuture> CreateExecutorValue(
      const v0::Value& value_pb) final {
    VLOG(2) << "Creating value: " << value_pb.Utf8DebugString();
    return ThreadRun(
        [value_pb, this]() -> absl::StatusOr<ExecutorValue> {
          return CreateValueAny(value_pb, this->mesh_, this->converter_.get());
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
          return fn->Call(arg, this->context_, this->dtensor_device_name_);
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
    TFF_TRY(value->MaterializeValue(context_, value_pb, dtensor_device_name_,
                                    tasks));
    TFF_TRY(tasks.WaitAll());
    return absl::OkStatus();
  }

 private:
  TFE_Context* context_ = nullptr;
  std::optional<std::string> dtensor_device_name_;
  int32_t max_concurrent_computation_calls_;
  std::optional<tensorflow::dtensor::Mesh> mesh_;
  std::unique_ptr<DTensorConverter> converter_;
  // ThreadPool should always be the last member so that in progress threads
  // with 'this' pointer are cleaned up before other members.
  ThreadPool thread_pool_;
};

}  // namespace

std::shared_ptr<Executor> CreateDTensorExecutor(
    TFE_Context* context, std::optional<std::string> dtensor_device_name,
    std::optional<tensorflow::dtensor::Mesh> mesh,
    std::unique_ptr<DTensorConverter> dtensor_converter,
    int32_t max_concurrent_computation_calls) {
  return std::make_shared<DTensorExecutor>(
      std::move(context), dtensor_device_name, mesh,
      dtensor_converter == nullptr ? std::make_unique<DTensorConverterImpl>()
                                   : std::move(dtensor_converter),
      max_concurrent_computation_calls);
}
}  // namespace tensorflow_federated

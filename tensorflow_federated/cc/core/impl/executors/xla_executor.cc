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

#include <future>  // NOLINT
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/tensor_serialization.h"
#include "tensorflow_federated/cc/core/impl/executors/threading.h"

namespace tensorflow_federated {

namespace {

// Representation of a tensor embedded in the XLA service. This class is
// responsible for owning the associated resources in the XLA service, and
// carrying sufficient information to materialize the tensor it represents into
// a TFF v0::Value.
class ServiceTensor {
 public:
  ServiceTensor(std::unique_ptr<xla::GlobalData> data,
                tensorflow::DataType dtype)
      : data_(std::move(data)), dtype_(dtype) {}

  tensorflow::DataType dtype() const { return dtype_; }
  xla::GlobalData* global_data() const { return data_.get(); }

 private:
  // XLA computations can be called with GlobalData* arguments, returning
  // GlobalData unique_ptrs. GlobalData represents an allocation of data in the
  // associated XLA service, so operating GlobalData-to-GlobalData in this way
  // minimizes transfers.
  std::unique_ptr<xla::GlobalData> data_;
  const tensorflow::DataType dtype_;
  // Since we hold a unique pointer internally, ServiceTensor is non-copyable
  // and non-copy-constructable.
  ServiceTensor(const ServiceTensor&) = delete;
  ServiceTensor& operator=(const ServiceTensor&) = delete;
};

// Representation for concrete values embedded in the XLA executor.
class XLAExecutorValue {
 public:
  enum class ValueType { TENSOR, UNIMPLEMENTED };

  explicit XLAExecutorValue(std::unique_ptr<xla::GlobalData> global_data,
                            tensorflow::DataType dtype)
      : value_(std::make_shared<ServiceTensor>(std::move(global_data), dtype)) {
  }

  ValueType type() const {
    if (absl::holds_alternative<std::shared_ptr<ServiceTensor>>(value_)) {
      return ValueType::TENSOR;
    } else {
      return ValueType::UNIMPLEMENTED;
    }
  }

  // Returns a pointer to the GlobalData backing an XLAExecutorValue of tensor
  // type. Requires that type() is ValueType::TENSOR. The pointer is guaranteed
  // to be valid as long as the XLAExecutorValue exists.
  std::shared_ptr<ServiceTensor> tensor() const {
    return absl::get<std::shared_ptr<ServiceTensor>>(value_);
  }

 private:
  using ValueVariant = absl::variant<std::shared_ptr<ServiceTensor>>;
  ValueVariant value_;
};

using ValueFuture =
    std::shared_future<absl::StatusOr<std::shared_ptr<XLAExecutorValue>>>;

class XLAExecutor : public ExecutorBase<ValueFuture> {
 public:
  explicit XLAExecutor(xla::Client* xla_client) : xla_client_(xla_client) {}

  absl::string_view ExecutorName() final { return "XLAExecutor"; }
  absl::StatusOr<ValueFuture> CreateExecutorValue(
      const v0::Value& value_pb) final {
    switch (value_pb.value_case()) {
      case v0::Value::ValueCase::kTensor: {
        return ThreadRun(
            [value_pb, this]() { return this->EmbedTensorValue(value_pb); });
      }
      default:
        return absl::UnimplementedError(absl::StrCat(
            "CreateValue not yet implemented for values of value case ",
            value_pb.value_case()));
    }
  }

  absl::StatusOr<ValueFuture> CreateCall(
      ValueFuture fn, absl::optional<ValueFuture> arg) final {
    return absl::UnimplementedError("Not implemented yet");
  }

  absl::StatusOr<ValueFuture> CreateStruct(
      std::vector<ValueFuture> members) final {
    return absl::UnimplementedError("Not implemented yet");
  }

  absl::StatusOr<ValueFuture> CreateSelection(ValueFuture value,
                                              const uint32_t index) final {
    return absl::UnimplementedError("Not implemented yet");
  }

  absl::Status Materialize(ValueFuture value, v0::Value* value_pb) final {
    std::shared_ptr<XLAExecutorValue> executor_value = TFF_TRY(Wait(value));
    switch (executor_value->type()) {
      case XLAExecutorValue::ValueType::TENSOR: {
        std::shared_ptr<ServiceTensor> tensor_in_service =
            executor_value->tensor();
        tensorflow::StatusOr<xla::Literal> result_literal =
            xla_client_->Transfer(*(tensor_in_service->global_data()));
        if (!result_literal.ok()) {
          return absl::InternalError(absl::StrCat(
              "Error transferring tensor from XLA service to host. Message: ",
              result_literal.status().error_message()));
        }
        tensorflow::Tensor tensor_out;
        tensorflow::Status tensor_conversion = tensorflow::LiteralToHostTensor(
            *result_literal, tensor_in_service->dtype(), &tensor_out);
        if (!tensor_conversion.ok()) {
          return absl::InternalError(
              absl::StrCat("Error converting XLA literal to tensor. Message: ",
                           tensor_conversion.error_message()));
        }
        TFF_TRY(SerializeTensorValue(tensor_out, value_pb));
        return absl::OkStatus();
      }
      default:
        return absl::UnimplementedError(
            absl::StrCat("Materialize only implemented for tensors. Attempted "
                         "to materialize a value of type ",
                         executor_value->type()));
    }
  }

 private:
  // Pointer to local XLA client. Assumed to be valid through the lifetime of
  // the executor.
  xla::Client* xla_client_;
  // Name specifying the platform which this executor will target. Assumed to be
  // registered in TensorFlow's MultiPlatformManager;.
  std::string platform_name_;

  absl::StatusOr<std::shared_ptr<XLAExecutorValue>> EmbedTensorValue(
      const v0::Value& value_pb) {
    tensorflow::Tensor t = TFF_TRY(DeserializeTensorValue(value_pb));
    xla::BorrowingLiteral tensor_literal;
    tensorflow::Status to_literal_status =
        tensorflow::HostTensorToBorrowingLiteral(t, &tensor_literal);
    if (!to_literal_status.ok()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Failed to convert v0::Value proto to XLA literal. Message: ",
          to_literal_status.error_message()));
    }
    tensorflow::StatusOr<std::unique_ptr<xla::GlobalData>> data_in_server =
        xla_client_->TransferToServer(tensor_literal);
    if (data_in_server.ok()) {
      return std::make_shared<XLAExecutorValue>(std::move(*data_in_server),
                                                t.dtype());
    } else {
      return absl::InvalidArgumentError(absl::StrCat(
          "Failed to transfer XLA literal to local server. Message: ",
          to_literal_status.error_message()));
    }
  }
};

absl::StatusOr<xla::Client*> GetXLAClient(absl::string_view platform_name) {
  tensorflow::StatusOr<xla::se::Platform*> platform =
      xla::se::MultiPlatformManager::PlatformWithName(platform_name);
  if (!platform.ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to find specified platform ", platform_name,
                     " in MultiPlatformManager. You may be missing a build "
                     "dependency to register the platform. Message: ",
                     platform.status().error_message()));
  }
  xla::LocalClientOptions options;
  options.set_platform(*platform);
  xla::StatusOr<xla::Client*> constructed_client =
      xla::ClientLibrary::GetOrCreateLocalClient(options);
  if (!constructed_client.ok()) {
    return absl::InternalError(
        absl::StrCat("Failed to construct XLA client. Message: ",
                     constructed_client.status().error_message()));
  }
  return *constructed_client;
}

}  // namespace

absl::StatusOr<std::shared_ptr<Executor>> CreateXLAExecutor(
    absl::string_view platform_name) {
  xla::Client* client = TFF_TRY(GetXLAClient(platform_name));
  return std::make_shared<XLAExecutor>(client);
}

}  // namespace tensorflow_federated

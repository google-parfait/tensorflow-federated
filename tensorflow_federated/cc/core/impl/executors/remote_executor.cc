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

#include "tensorflow_federated/cc/core/impl/executors/remote_executor.h"

#include <future>  // NOLINT
#include <memory>
#include <variant>

#include "net/grpc/public/include/grpcpp/impl/codegen/channel_interface.h"
#include "net/grpc/public/include/grpcpp/impl/codegen/client_context.h"
#include "net/grpc/public/include/grpcpp/impl/codegen/status.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/threading.h"
#include "tensorflow_federated/proto/v0/computation.proto.h"
#include "tensorflow_federated/proto/v0/executor.grpc.pb.h"
#include "tensorflow_federated/proto/v0/executor.proto.h"

namespace tensorflow_federated {

class ExecutorValue;

using ValueFuture =
    std::shared_future<absl::StatusOr<std::shared_ptr<ExecutorValue>>>;

class RemoteExecutor : public ExecutorBase<ValueFuture> {
 public:
  RemoteExecutor(std::unique_ptr<v0::Executor::StubInterface> stub,
                 const CardinalityMap& cardinalities)
      : stub_(std::move(stub)), cardinalities_(cardinalities) {}
  ~RemoteExecutor() {}

  const char* ExecutorName() final { return "RemoteExecutor"; }

  absl::StatusOr<ValueFuture> CreateExecutorValue(
      const v0::Value& value_pb) final;

  absl::StatusOr<ValueFuture> CreateCall(
      ValueFuture function, std::optional<ValueFuture> argument) final;

  absl::StatusOr<ValueFuture> CreateStruct(
      std::vector<ValueFuture> members) final;

  absl::StatusOr<ValueFuture> CreateSelection(ValueFuture value,
                                              const uint32_t index) final;

  absl::Status Materialize(ValueFuture value, v0::Value* value_pb) final;

 private:
  grpc::Status EnsureInitialized();
  std::shared_ptr<v0::Executor::StubInterface> stub_;
  absl::Mutex mutex_;
  bool cardinalities_set_ ABSL_GUARDED_BY(mutex_) = false;
  CardinalityMap cardinalities_;
};

class ExecutorValue {
 public:
  ExecutorValue(const v0::ValueRef&& value_ref,
                std::shared_ptr<v0::Executor::StubInterface> stub)
      : value_ref_(value_ref), stub_(stub) {}
  // Dispose implemented for now just on destructors, and stub is copied in.
  ~ExecutorValue() {
    ThreadRun([dispose_id = value_ref_.id(), stub = stub_] {
      v0::DisposeRequest request;
      v0::DisposeResponse response;
      grpc::ClientContext context;
      v0::ValueRef value_to_dispose;
      *value_to_dispose.mutable_id() = dispose_id;
      request.mutable_value_ref()->Add(std::move(value_to_dispose));
      absl::Status dispose_status = stub->Dispose(&context, request, &response);
      if (!dispose_status.ok()) {
        LOG(ERROR) << "Error disposing of ExecutorValue [" << dispose_id
                   << "]: " << dispose_status;
      }
    });
  }

  v0::ValueRef Get() const { return value_ref_; }

 private:
  v0::ValueRef value_ref_;
  std::shared_ptr<v0::Executor::StubInterface> stub_;
};

grpc::Status RemoteExecutor::EnsureInitialized() {
  absl::MutexLock lock(&mutex_);
  if (cardinalities_set_) {
    return absl::OkStatus();
  }
  v0::SetCardinalitiesRequest request;
  for (auto iter = cardinalities_.begin(); iter != cardinalities_.end();
       ++iter) {
    v0::Placement placement;
    placement.set_uri(iter->first);
    v0::SetCardinalitiesRequest::Cardinality cardinality;
    *cardinality.mutable_placement() = placement;
    cardinality.set_cardinality(iter->second);
    request.mutable_cardinalities()->Add(std::move(cardinality));
  }
  v0::SetCardinalitiesResponse response;
  grpc::ClientContext client_context;
  auto result = stub_->SetCardinalities(&client_context, request, &response);
  if (result.ok()) {
    cardinalities_set_ = true;
  }
  return result;
}

absl::StatusOr<ValueFuture> RemoteExecutor::CreateExecutorValue(
    const v0::Value& value_pb) {
  TFF_TRY(EnsureInitialized());
  v0::CreateValueRequest request;
  request.mutable_value()->CopyFrom(value_pb);

  return ThreadRun(
      [request = std::move(request),
       stub = this->stub_]() -> absl::StatusOr<std::shared_ptr<ExecutorValue>> {
        v0::CreateValueResponse response;
        grpc::ClientContext client_context;
        grpc::Status status =
            stub->CreateValue(&client_context, request, &response);
        if (!status.ok()) {
          return status;
        }
        return std::make_shared<ExecutorValue>(std::move(response.value_ref()),
                                               stub);
      });
}

absl::StatusOr<ValueFuture> RemoteExecutor::CreateCall(
    ValueFuture function, std::optional<ValueFuture> argument) {
  TFF_TRY(EnsureInitialized());
  return ThreadRun(
      [function = std::move(function), argument = std::move(argument),
       stub = this->stub_]() -> absl::StatusOr<std::shared_ptr<ExecutorValue>> {
        v0::CreateCallRequest request;
        v0::CreateCallResponse response;
        grpc::ClientContext context;
        std::shared_ptr<ExecutorValue> fn = TFF_TRY(Wait(function));

        *request.mutable_function_ref() = fn->Get();
        if (argument.has_value()) {
          std::shared_ptr<ExecutorValue> arg_value =
              TFF_TRY(Wait(argument.value()));
          *request.mutable_argument_ref() = arg_value->Get();
        }

        grpc::Status status = stub->CreateCall(&context, request, &response);
        if (!status.ok()) {
          return status;
        }
        return std::make_shared<ExecutorValue>(std::move(response.value_ref()),
                                               stub);
      });
}

absl::StatusOr<ValueFuture> RemoteExecutor::CreateStruct(
    std::vector<ValueFuture> members) {
  TFF_TRY(EnsureInitialized());
  return ThreadRun(
      [futures = std::move(members),
       stub = this->stub_]() -> absl::StatusOr<std::shared_ptr<ExecutorValue>> {
        v0::CreateStructRequest request;
        v0::CreateStructResponse response;
        grpc::ClientContext context;
        std::vector<std::shared_ptr<ExecutorValue>> values =
            TFF_TRY(WaitAll(futures));
        for (const std::shared_ptr<ExecutorValue>& element : values) {
          v0::CreateStructRequest_Element struct_elem;
          *struct_elem.mutable_value_ref() = element->Get();
          request.mutable_element()->Add(std::move(struct_elem));
        }
        grpc::Status status = stub->CreateStruct(&context, request, &response);
        if (!status.ok()) {
          return status;
        }
        return std::make_shared<ExecutorValue>(std::move(response.value_ref()),
                                               stub);
      });
}

absl::StatusOr<ValueFuture> RemoteExecutor::CreateSelection(
    ValueFuture value, const uint32_t index) {
  TFF_TRY(EnsureInitialized());
  return ThreadRun(
      [source = std::move(value), index = index,
       stub = this->stub_]() -> absl::StatusOr<std::shared_ptr<ExecutorValue>> {
        v0::CreateSelectionRequest request;
        v0::CreateSelectionResponse response;
        grpc::ClientContext context;
        std::shared_ptr<ExecutorValue> source_value = TFF_TRY(Wait(source));
        *request.mutable_source_ref() = source_value->Get();
        request.set_index(index);
        grpc::Status status =
            stub->CreateSelection(&context, request, &response);
        if (!status.ok()) {
          return status;
        }
        return std::make_shared<ExecutorValue>(std::move(response.value_ref()),
                                               stub);
      });
}

absl::Status RemoteExecutor::Materialize(ValueFuture value,
                                         v0::Value* value_pb) {
  v0::ComputeRequest request;
  std::shared_ptr<ExecutorValue> value_ref = TFF_TRY(Wait(value));
  *request.mutable_value_ref() = value_ref->Get();

  v0::ComputeResponse compute_response;
  grpc::ClientContext client_context;
  grpc::Status status =
      stub_->Compute(&client_context, request, &compute_response);
  *value_pb = std::move(*compute_response.mutable_value());
  return status;
}

std::shared_ptr<Executor> CreateRemoteExecutor(
    std::unique_ptr<v0::Executor::StubInterface> stub,
    const CardinalityMap& cardinalities) {
  return std::make_shared<RemoteExecutor>(std::move(stub), cardinalities);
}

std::shared_ptr<Executor> CreateRemoteExecutor(
    std::shared_ptr<grpc::ChannelInterface> channel,
    const CardinalityMap& cardinalities) {
  std::unique_ptr<v0::Executor::StubInterface> stub(
      v0::Executor::NewStub(channel));
  return std::make_shared<RemoteExecutor>(std::move(stub), cardinalities);
}
}  // namespace tensorflow_federated

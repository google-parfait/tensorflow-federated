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

#include <cstdint>
#include <future>  // NOLINT
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/grpcpp.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/status_conversion.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/threading.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"
#include "tensorflow_federated/proto/v0/executor.grpc.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

class ExecutorValue;

using ValueFuture =
    std::shared_future<absl::StatusOr<std::shared_ptr<ExecutorValue>>>;

// A custom deleter for the `std::shared_ptr<v0::ExecutorGroup::StubInterface>`
// which will call `DisposeExecutor` for the provided `executor_pb`, if any.
// This ensures that the remote service knows no more calls will be coming for
// the given `executor_pb` and that the associated resources can be released.
//
// Unfortunately, this cannot be part of the destructor of `RemoteExecutor`, as
// `RemoteExecutor`'s methods may start threads that send new GRPC requests for
// the given executor after the `RemoteExecutor` has already been destroyed.
class StubDeleter {
 public:
  StubDeleter() = default;
  void SetExecutorId(v0::ExecutorId executor_pb) {
    executor_pb_ = std::move(executor_pb);
  }
  void operator()(v0::ExecutorGroup::StubInterface* stub) {
    if (executor_pb_.has_value()) {
      ThreadRun([stub, executor_pb = std::move(*executor_pb_)]() {
        v0::DisposeExecutorRequest request;
        v0::DisposeExecutorResponse response;
        grpc::ClientContext context;
        *request.mutable_executor() = std::move(executor_pb);
        grpc::Status dispose_status =
            stub->DisposeExecutor(&context, request, &response);
        if (!dispose_status.ok()) {
          LOG(ERROR) << "Error disposing of Executor ["
                     << request.executor().id()
                     << "]: " << grpc_to_absl(dispose_status);
        }
        delete stub;
      });
    } else {
      delete stub;
    }
  }

 private:
  std::optional<v0::ExecutorId> executor_pb_;
};

class RemoteExecutor : public ExecutorBase<ValueFuture> {
 public:
  RemoteExecutor(std::unique_ptr<v0::ExecutorGroup::StubInterface> stub,
                 const CardinalityMap& cardinalities,
                 const bool stream_structs = false)
      : stub_(stub.release(), StubDeleter()),
        cardinalities_(cardinalities),
        stream_structs_(stream_structs) {}

  ~RemoteExecutor() override = default;

  std::string_view ExecutorName() final {
    static constexpr std::string_view kExecutorName = "RemoteExecutor";
    return kExecutorName;
  }

  absl::StatusOr<ValueFuture> CreateExecutorValue(
      const v0::Value& value_pb) final;

  absl::StatusOr<ValueFuture> CreateCall(
      ValueFuture function, std::optional<ValueFuture> argument) final;

  absl::StatusOr<ValueFuture> CreateStruct(
      std::vector<ValueFuture> members) final;

  absl::StatusOr<ValueFuture> CreateSelection(ValueFuture value,
                                              uint32_t index) final;

  absl::Status Materialize(ValueFuture value, v0::Value* value_pb) final;

 private:
  absl::Status EnsureInitialized();
  std::shared_ptr<v0::ExecutorGroup::StubInterface> stub_;
  CardinalityMap cardinalities_;
  absl::Mutex mutex_;
  bool executor_pb_set_ ABSL_GUARDED_BY(mutex_) = false;
  v0::ExecutorId executor_pb_;
  const bool stream_structs_ = false;
  absl::Mutex stream_structs_map_mutex_;
  std::unordered_map<std::string, int32_t> stream_structs_map_ ABSL_GUARDED_BY(
      stream_structs_map_mutex_);  // map of struct or a computation
                                   // with struct return type to its number of
                                   // elements. Populated only when
                                   // stream_structs_ is true.
  int32_t NumElements(
      const v0::ValueRef& value_ref);  // returns 0 if stream_structs_ is false
  void SetNumElements(
      const v0::ValueRef& value_ref,
      int32_t num_elements);  // populates the stream_structs_map_, only when
                              // stream_structs_ is true;
};

class ExecutorValue {
 public:
  ExecutorValue(v0::ValueRef value_ref, v0::ExecutorId executor_pb,
                std::shared_ptr<v0::ExecutorGroup::StubInterface> stub)
      : value_ref_(std::move(value_ref)),
        executor_pb_(std::move(executor_pb)),
        stub_(stub) {}
  // Dispose implemented for now just on destructors, and stub is copied in.
  ~ExecutorValue() {
    ThreadRun([value_ref = value_ref_, executor_pb = executor_pb_,
               stub = stub_] {
      v0::DisposeRequest request;
      v0::DisposeResponse response;
      grpc::ClientContext context;
      *request.mutable_executor() = std::move(executor_pb);
      *request.add_value_ref() = value_ref;
      grpc::Status dispose_status = stub->Dispose(&context, request, &response);
      if (!dispose_status.ok()) {
        LOG(ERROR) << "Error disposing of ExecutorValue [" << value_ref.id()
                   << "]: " << grpc_to_absl(dispose_status);
      }
    });
  }

  v0::ValueRef Get() const { return value_ref_; }

 private:
  v0::ValueRef value_ref_;
  v0::ExecutorId executor_pb_;
  std::shared_ptr<v0::ExecutorGroup::StubInterface> stub_;
};

absl::Status RemoteExecutor::EnsureInitialized() {
  absl::MutexLock lock(&mutex_);
  if (executor_pb_set_) {
    return absl::OkStatus();
  }
  v0::GetExecutorRequest request;
  for (auto iter = cardinalities_.begin(); iter != cardinalities_.end();
       ++iter) {
    v0::Placement placement;
    placement.set_uri(iter->first);
    v0::Cardinality cardinality;
    *cardinality.mutable_placement() = placement;
    cardinality.set_cardinality(iter->second);
    request.mutable_cardinalities()->Add(std::move(cardinality));
  }
  v0::GetExecutorResponse response;
  grpc::ClientContext client_context;
  auto result = stub_->GetExecutor(&client_context, request, &response);
  if (result.ok()) {
    executor_pb_ = response.executor();
    executor_pb_set_ = true;
    // Tell the `StubDeleter` which executor it should delete when the stub is
    // no longer referenced.
    std::get_deleter<StubDeleter>(stub_)->SetExecutorId(executor_pb_);
  }
  return grpc_to_absl(result);
}

int32_t RemoteExecutor::NumElements(const v0::ValueRef& value_ref) {
  if (!stream_structs_) {
    return 0;
  }
  absl::ReaderMutexLock lock(&stream_structs_map_mutex_);
  auto it = stream_structs_map_.find(value_ref.id());
  return it == stream_structs_map_.end() ? 0 : it->second;
}

void RemoteExecutor::SetNumElements(const v0::ValueRef& value_ref,
                                    int32_t num_elements) {
  if (stream_structs_) {
    absl::WriterMutexLock lock(&stream_structs_map_mutex_);
    stream_structs_map_[value_ref.id()] = num_elements;
  }
}

absl::StatusOr<ValueFuture> RemoteExecutor::CreateExecutorValue(
    const v0::Value& value_pb) {
  TFF_TRY(EnsureInitialized());
  // b/265163585 : Extend for federated values.
  if (stream_structs_ && value_pb.has_struct_()) {
    const auto& struct_pb = value_pb.struct_();
    std::vector<ValueFuture> elements;
    for (int i = 0; i < struct_pb.element_size(); ++i) {
      const v0::Value::Struct::Element& element_pb = struct_pb.element(i);
      auto element = CreateExecutorValue(element_pb.value());
      if (!element.ok()) {
        return element.status();
      }
      elements.push_back(std::move(element.value()));
    }

    auto result = CreateStruct(std::move(elements));
    if (!result.ok()) {
      return result.status();
    }
    std::shared_ptr<ExecutorValue> value_ref = TFF_TRY(Wait(result.value()));
    SetNumElements(value_ref->Get(), struct_pb.element_size());
    return result;
  }

  v0::CreateValueRequest request;
  *request.mutable_executor() = executor_pb_;
  *request.mutable_value() = value_pb;

  return ThreadRun([request = std::move(request), executor_pb = executor_pb_,
                    this, this_keepalive = shared_from_this()]()
                       -> absl::StatusOr<std::shared_ptr<ExecutorValue>> {
    v0::CreateValueResponse response;
    grpc::ClientContext client_context;
    grpc::Status status =
        this->stub_->CreateValue(&client_context, request, &response);
    TFF_TRY(grpc_to_absl(status));
    return std::make_shared<ExecutorValue>(std::move(response.value_ref()),
                                           executor_pb, this->stub_);
  });
}

absl::StatusOr<ValueFuture> RemoteExecutor::CreateCall(
    ValueFuture function, std::optional<ValueFuture> argument) {
  TFF_TRY(EnsureInitialized());
  return ThreadRun([function = std::move(function),
                    argument = std::move(argument), executor_pb = executor_pb_,
                    this, this_keepalive = shared_from_this()]()
                       -> absl::StatusOr<std::shared_ptr<ExecutorValue>> {
    v0::CreateCallRequest request;
    v0::CreateCallResponse response;
    grpc::ClientContext context;
    std::shared_ptr<ExecutorValue> fn = TFF_TRY(Wait(function));

    *request.mutable_executor() = executor_pb;
    *request.mutable_function_ref() = fn->Get();
    if (argument.has_value()) {
      std::shared_ptr<ExecutorValue> arg_value =
          TFF_TRY(Wait(argument.value()));
      *request.mutable_argument_ref() = arg_value->Get();
    }

    grpc::Status status = this->stub_->CreateCall(&context, request, &response);
    TFF_TRY(grpc_to_absl(status));
    auto result = std::make_shared<ExecutorValue>(
        std::move(response.value_ref()), executor_pb, this->stub_);
    if (NumElements(fn->Get()) > 0) {
      SetNumElements(result->Get(), NumElements(fn->Get()));
    }
    return result;
  });
}

absl::StatusOr<ValueFuture> RemoteExecutor::CreateStruct(
    std::vector<ValueFuture> members) {
  TFF_TRY(EnsureInitialized());
  return ThreadRun([futures = std::move(members), this,
                    this_keepalive = shared_from_this()]()
                       -> absl::StatusOr<std::shared_ptr<ExecutorValue>> {
    v0::CreateStructRequest request;
    *request.mutable_executor() = this->executor_pb_;
    v0::CreateStructResponse response;
    grpc::ClientContext context;
    std::vector<std::shared_ptr<ExecutorValue>> values =
        TFF_TRY(WaitAll(futures));
    for (const std::shared_ptr<ExecutorValue>& element : values) {
      v0::CreateStructRequest_Element struct_elem;
      *struct_elem.mutable_value_ref() = element->Get();
      request.mutable_element()->Add(std::move(struct_elem));
    }
    grpc::Status status =
        this->stub_->CreateStruct(&context, request, &response);
    TFF_TRY(grpc_to_absl(status));
    auto result = std::make_shared<ExecutorValue>(
        std::move(response.value_ref()), this->executor_pb_, this->stub_);
    SetNumElements(result->Get(), futures.size());
    return result;
  });
}

absl::StatusOr<ValueFuture> RemoteExecutor::CreateSelection(
    ValueFuture value, const uint32_t index) {
  TFF_TRY(EnsureInitialized());
  return ThreadRun([source = std::move(value), index = index,
                    executor_pb = executor_pb_, this,
                    this_keepalive = shared_from_this()]()
                       -> absl::StatusOr<std::shared_ptr<ExecutorValue>> {
    v0::CreateSelectionRequest request;
    v0::CreateSelectionResponse response;
    grpc::ClientContext context;
    std::shared_ptr<ExecutorValue> source_value = TFF_TRY(Wait(source));
    *request.mutable_executor() = executor_pb;
    *request.mutable_source_ref() = source_value->Get();
    request.set_index(index);
    grpc::Status status =
        this->stub_->CreateSelection(&context, request, &response);
    TFF_TRY(grpc_to_absl(status));
    return std::make_shared<ExecutorValue>(std::move(response.value_ref()),
                                           executor_pb, this->stub_);
  });
}

absl::Status RemoteExecutor::Materialize(ValueFuture value,
                                         v0::Value* value_pb) {
  std::shared_ptr<ExecutorValue> value_ref = TFF_TRY(Wait(value));
  if (NumElements(value_ref->Get()) > 0) {
    v0::Value::Struct* struct_pb = value_pb->mutable_struct_();
    for (int i = 0; i < NumElements(value_ref->Get()); i++) {
      const auto selection = TFF_TRY(CreateSelection(value, i));
      v0::Value element_value;
      TFF_TRY(Materialize(selection, &element_value));
      *(struct_pb->add_element()->mutable_value()) = element_value;
    }
    return absl::OkStatus();
  }

  v0::ComputeRequest request;
  *request.mutable_executor() = executor_pb_;
  *request.mutable_value_ref() = value_ref->Get();

  v0::ComputeResponse compute_response;
  grpc::ClientContext client_context;
  grpc::Status status =
      stub_->Compute(&client_context, request, &compute_response);
  *value_pb = std::move(*compute_response.mutable_value());
  return grpc_to_absl(status);
}

std::shared_ptr<Executor> CreateRemoteExecutor(
    std::unique_ptr<v0::ExecutorGroup::StubInterface> stub,
    const CardinalityMap& cardinalities, const bool stream_structs) {
  return std::make_shared<RemoteExecutor>(std::move(stub), cardinalities,
                                          stream_structs);
}

std::shared_ptr<Executor> CreateRemoteExecutor(
    std::shared_ptr<grpc::ChannelInterface> channel,
    const CardinalityMap& cardinalities, const bool stream_structs) {
  std::unique_ptr<v0::ExecutorGroup::StubInterface> stub(
      v0::ExecutorGroup::NewStub(channel));
  return std::make_shared<RemoteExecutor>(std::move(stub), cardinalities,
                                          stream_structs);
}
}  // namespace tensorflow_federated

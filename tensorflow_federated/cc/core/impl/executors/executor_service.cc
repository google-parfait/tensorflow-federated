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

#include "tensorflow_federated/cc/core/impl/executors/executor_service.h"

#include <grpcpp/support/status.h>

#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

#define TFF_TRYLOG_GRPC(status)               \
  ({                                          \
    grpc::Status __status = status;           \
    if (!__status.ok()) {                     \
      LOG(ERROR) << __status.error_message(); \
      return __status;                        \
    }                                         \
  })

namespace {

// Creates a unique string for cardinalities like "CLIENTS=4,SERVER=1"
std::string CardinalitiesToString(const CardinalityMap& cardinalities) {
  return absl::StrJoin(cardinalities, ",", absl::PairFormatter("="));
}

v0::ValueRef IdToRemoteValue(ValueId value_id) {
  v0::ValueRef value_ref;
  value_ref.set_id(absl::StrCat(value_id));
  return value_ref;
}

grpc::Status RemoteValueToId(const v0::ValueRef& remote_value_ref,
                             ValueId& value_id_out) {
  // Incoming ref should be a string containing the ValueId.
  if (absl::SimpleAtoi(remote_value_ref.id(), &value_id_out)) {
    return grpc::Status::OK;
  }
  return grpc::Status(
      grpc::StatusCode::INVALID_ARGUMENT,
      absl::StrCat("Expected value ref to be an integer id, found ",
                   remote_value_ref.id()));
}

}  // namespace

grpc::Status ExecutorService::GetExecutor(grpc::ServerContext* context,
                                          const v0::GetExecutorRequest* request,
                                          v0::GetExecutorResponse* response) {
  CardinalityMap cardinalities;
  for (const auto& cardinality : request->cardinalities()) {
    cardinalities.insert(
        {cardinality.placement().uri(), cardinality.cardinality()});
  }
  std::string executor_key = CardinalitiesToString(cardinalities);
  {
    absl::WriterMutexLock lock(&executors_mutex_);
    auto entry = executors_.find(executor_key);
    bool executor_exists = (entry != executors_.end());
    if (!executor_exists) {
      // Initialize the `std::shared_ptr<Executor>` added to the map.
      absl::StatusOr<std::shared_ptr<Executor>> new_executor =
          executor_factory_(cardinalities);
      if (!new_executor.ok()) {
        LOG(ERROR) << "Failure to construct executor in executor service: ";
        LOG(ERROR) << new_executor.status().message();
        return absl_to_grpc(new_executor.status());
      }
      // Initialize the refcount to one.
      executors_.insert(
          {executor_key, ExecutorEntry{std::move(*new_executor), 1}});
      LOG(INFO) << "ExecutorService created new Executor for cardinalities: "
                << executor_key;
    } else {
      // Just increment the refcount of the entry.
      entry->second.remote_refcount++;
    }
  }
  response->mutable_executor()->set_id(executor_key);
  return grpc::Status::OK;
}

grpc::Status ExecutorService::RequireExecutor(
    absl::string_view method_name, const v0::ExecutorId& executor,
    std::shared_ptr<Executor>& executor_out) {
  absl::ReaderMutexLock lock(&executors_mutex_);
  auto it = executors_.find(executor.id());
  if (it != executors_.end()) {
    executor_out = it->second.executor;
    return grpc::Status::OK;
  }
  // A lack of executor in the expected slot is retryable, but clients must
  // ensure the service state is adjusted (e.g. with a GetExecutor call) before
  // retrying. Following
  // https://grpc.github.io/grpc/core/md_doc_statuscodes.html we raise
  // FailedPrecondition.
  return grpc::Status(
      grpc::StatusCode::FAILED_PRECONDITION,
      absl::StrCat("Error evaluating `ExecutorService::", method_name,
                   "`. No executor found for ID: '", executor.id(), "'."));
}

void ExecutorService::DestroyExecutor(const v0::ExecutorId& executor) {
  absl::WriterMutexLock lock(&executors_mutex_);
  int elements_erased = executors_.erase(executor.id());
  if (elements_erased > 1) {
    VLOG(2) << "More elements erased than expected while destroying executor; "
               "expected 0 or 1, erased "
            << elements_erased;
  }
}

grpc::Status ExecutorService::HandleNotOK(const absl::Status& status,
                                          const v0::ExecutorId& executor_id) {
  if (status.code() == absl::StatusCode::kFailedPrecondition) {
    VLOG(1) << "Destroying executor " << executor_id.Utf8DebugString();
    DestroyExecutor(executor_id);
  }
  VLOG(1) << status.message();
  return absl_to_grpc(status);
}

grpc::Status ExecutorService::CreateValue(grpc::ServerContext* context,
                                          const v0::CreateValueRequest* request,
                                          v0::CreateValueResponse* response) {
  std::shared_ptr<Executor> executor;
  TFF_TRYLOG_GRPC(
      RequireExecutor("CreateValue", request->executor(), executor));
  absl::StatusOr<OwnedValueId> id = executor->CreateValue(request->value());
  if (!id.ok()) {
    return HandleNotOK(id.status(), request->executor());
  }
  *response->mutable_value_ref() = IdToRemoteValue(id.value());
  // We must call forget on the embedded id to prevent the destructor from
  // running when the variable goes out of scope. Similar considerations apply
  // to the reset of the Create methods below.
  id.value().forget();
  return grpc::Status::OK;
}

grpc::Status ExecutorService::CreateCall(grpc::ServerContext* context,
                                         const v0::CreateCallRequest* request,
                                         v0::CreateCallResponse* response) {
  std::shared_ptr<Executor> executor;
  TFF_TRYLOG_GRPC(RequireExecutor("CreateCall", request->executor(), executor));
  ValueId embedded_fn;
  TFF_TRYLOG_GRPC(RemoteValueToId(request->function_ref(), embedded_fn));
  absl::optional<ValueId> embedded_arg;
  if (request->has_argument_ref()) {
    embedded_arg = 0;
    TFF_TRYLOG_GRPC(
        RemoteValueToId(request->argument_ref(), embedded_arg.value()));
  }
  absl::StatusOr<OwnedValueId> called_fn =
      executor->CreateCall(embedded_fn, embedded_arg);
  if (!called_fn.ok()) {
    return HandleNotOK(called_fn.status(), request->executor());
  }
  *response->mutable_value_ref() = IdToRemoteValue(called_fn.value());
  // We must prevent this destructor from running similarly to CreateValue.
  called_fn.value().forget();
  return grpc::Status::OK;
}

grpc::Status ExecutorService::CreateStruct(
    grpc::ServerContext* context, const v0::CreateStructRequest* request,
    v0::CreateStructResponse* response) {
  std::shared_ptr<Executor> executor;
  TFF_TRYLOG_GRPC(
      RequireExecutor("CreateStruct", request->executor(), executor));
  std::vector<ValueId> requested_ids;
  requested_ids.reserve(request->element().size());
  for (const v0::CreateStructRequest::Element& elem : request->element()) {
    ValueId id;
    TFF_TRYLOG_GRPC(RemoteValueToId(elem.value_ref(), id));
    requested_ids.push_back(id);
  }
  absl::StatusOr<OwnedValueId> created_struct =
      executor->CreateStruct(requested_ids);
  if (!created_struct.ok()) {
    return HandleNotOK(created_struct.status(), request->executor());
  }
  *response->mutable_value_ref() = IdToRemoteValue(created_struct.value());
  // We must prevent this destructor from running similarly to CreateValue.
  created_struct.value().forget();
  return grpc::Status::OK;
}

grpc::Status ExecutorService::CreateSelection(
    grpc::ServerContext* context, const v0::CreateSelectionRequest* request,
    v0::CreateSelectionResponse* response) {
  std::shared_ptr<Executor> executor;
  TFF_TRYLOG_GRPC(
      RequireExecutor("CreateSelection", request->executor(), executor));
  ValueId selection_source;
  TFF_TRYLOG_GRPC(RemoteValueToId(request->source_ref(), selection_source));
  absl::StatusOr<OwnedValueId> selected_element =
      executor->CreateSelection(selection_source, request->index());
  if (!selected_element.ok()) {
    return HandleNotOK(selected_element.status(), request->executor());
  }
  *response->mutable_value_ref() = IdToRemoteValue(selected_element.value());
  // We must prevent this destructor from running similarly to CreateValue.
  selected_element.value().forget();
  return grpc::Status::OK;
}

grpc::Status ExecutorService::Compute(grpc::ServerContext* context,
                                      const v0::ComputeRequest* request,
                                      v0::ComputeResponse* response) {
  std::shared_ptr<Executor> executor;
  TFF_TRYLOG_GRPC(RequireExecutor("Compute", request->executor(), executor));
  ValueId requested_value;
  TFF_TRYLOG_GRPC(RemoteValueToId(request->value_ref(), requested_value));
  absl::Status status =
      executor->Materialize(requested_value, response->mutable_value());
  if (status.ok()) {
    return absl_to_grpc(status);
  }
  return HandleNotOK(status, request->executor());
}

grpc::Status ExecutorService::Dispose(grpc::ServerContext* context,
                                      const v0::DisposeRequest* request,
                                      v0::DisposeResponse* response) {
  std::shared_ptr<Executor> executor;
  TFF_TRYLOG_GRPC(RequireExecutor("Dispose", request->executor(), executor));
  std::vector<ValueId> embedded_ids_to_dispose;
  embedded_ids_to_dispose.reserve(request->value_ref().size());
  // Filter the requested IDs to those corresponding to the currently live
  // executors. Clients are free to batch these requests however they want or
  // let them be called by a GC mechanism, so we do not force the client to only
  // dispose of values in the live executor.
  for (const v0::ValueRef& disposed_value_ref : request->value_ref()) {
    ValueId embedded_value;
    grpc::Status status = RemoteValueToId(disposed_value_ref, embedded_value);
    if (status.ok()) {
      absl::Status absl_status = executor->Dispose(embedded_value);
      if (!absl_status.ok()) {
        LOG(ERROR) << absl_status.message();
        return absl_to_grpc(absl_status);
      }
    }
  }
  return grpc::Status::OK;
}

grpc::Status ExecutorService::DisposeExecutor(
    grpc::ServerContext* context, const v0::DisposeExecutorRequest* request,
    v0::DisposeExecutorResponse* response) {
  absl::WriterMutexLock lock(&executors_mutex_);
  const std::string& id = request->executor().id();
  auto it = executors_.find(id);
  if (it != executors_.end()) {
    ExecutorEntry& entry = it->second;
    entry.remote_refcount--;
    if (entry.remote_refcount == 0) {
      executors_.erase(it);
    }
    return grpc::Status::OK;
  } else {
    // TODO(b/183942515): This case is expected to be potentially hit during
    // normal execution. Perhaps we can think of a better manner of handling to
    // avoid spamming client logs.
    return grpc::Status(
        grpc::StatusCode::INVALID_ARGUMENT,
        absl::StrCat("Error evaluating `ExecutorService::DisposeExecutor`. ",
                     "No executor found for ID: '", id, "'."));
  }
}

}  // namespace tensorflow_federated

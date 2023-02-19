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

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/status_conversion.h"
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

using ExecutorId = std::string;

absl::StatusOr<ExecutorId>
ExecutorService::ExecutorResolver::ExecutorIDForRequirements(
    const ExecutorRequirements& requirements) {
  // TODO(b/236264677): This direct executor caching can cause unevenly
  // partitioned work in the presence of aggregators. The TFF RemoteExecutor
  // currently proxies for a physical (as opposed to logical) machine; therefore
  // this caching essentially 'freezes' the remote machines available at the
  // time of executor construction as the only ones receiving work, if this
  // service represents an aggregation service.
  absl::MutexLock lock(&executors_mutex_);
  std::string cardinalities_string =
      CardinalitiesToString(requirements.cardinalities);
  auto it = executors_.find(cardinalities_string);
  bool executor_exists = (it != executors_.end());
  if (!executor_exists) {
    absl::StatusOr<std::shared_ptr<Executor>> new_executor =
        ex_factory_(requirements.cardinalities);
    if (!new_executor.ok()) {
      LOG(ERROR) << "Failure to construct executor in executor service: ";
      LOG(ERROR) << new_executor.status().message();
      return new_executor.status();
    }
    // Give the executor a unique key, incrementing the executor index so that
    // the next construction call yields another unique ID.
    std::string executor_key = absl::StrCat(
        cardinalities_string, "/", service_id_, "/", executor_index_++);
    // Ensure keys_to_cardinalities_ key corresponds to an executor in
    // executors_.
    keys_to_cardinalities_.emplace(executor_key, cardinalities_string);
    // Initialize the refcount to one, and the ID to the one constructed above.
    ExecutorEntry entry({std::move(*new_executor), 1, executor_key});
    executors_.emplace(cardinalities_string, entry);
    VLOG(2) << "ExecutorService created new Executor for cardinalities: "
            << cardinalities_string;
    VLOG(2) << "Returning to clients executor ID: " << executor_key;
    return entry.executor_id;
  } else {
    // Just increment the refcount of the entry.
    it->second.remote_refcount++;
    ExecutorEntry entry = it->second;
    return entry.executor_id;
  }
}

absl::StatusOr<ExecutorService::ExecutorEntry>
ExecutorService::ExecutorResolver::ExecutorForId(const ExecutorId& ex_id) {
  absl::MutexLock lock(&executors_mutex_);
  auto ex_cardinalities_it = keys_to_cardinalities_.find(ex_id);
  if (ex_cardinalities_it == keys_to_cardinalities_.end()) {
    // A lack of executor in the expected slot is retryable, but clients must
    // ensure the service state is adjusted (e.g. with a GetExecutor call)
    // before retrying. Following
    // https://grpc.github.io/grpc/core/md_doc_statuscodes.html we raise
    // FailedPrecondition.
    return absl::FailedPreconditionError(
        absl::StrCat("No executor found for ID: '", ex_id, "'."));
  }
  auto ex_it = executors_.find(ex_cardinalities_it->second);
  if (ex_it == executors_.end()) {
    return absl::InternalError(absl::StrCat(
        "No executor found for cardinalities string: ",
        ex_cardinalities_it->second, ", referred to by executor id ", ex_id));
  }
  return ex_it->second;
}

absl::Status ExecutorService::ExecutorResolver::DisposeExecutor(
    const ExecutorId& ex_id) {
  bool should_destroy;
  // We take a writer lock here because we must decrement the refcount.
  absl::MutexLock lock(&executors_mutex_);
  auto ex_cardinalities_it = keys_to_cardinalities_.find(ex_id);
  if (ex_cardinalities_it == keys_to_cardinalities_.end()) {
    // DisposeExecutor can occur on a deleted executor in the case of a worker
    // failure, since Python GC will trigger a DisposeExecutor call while the
    // execution context attempts to retry the call. We may, however, want to
    // rather 'mark' executors deleted in this manner, so that
    // double-DisposeExecutor does not pass, as that indicates a potential
    // client-side bug.
    return absl::OkStatus();
  }
  auto ex_it = executors_.find(ex_cardinalities_it->second);
  if (ex_it == executors_.end()) {
    return absl::InternalError(absl::StrCat(
        "No executor found for cardinalities string: ",
        ex_cardinalities_it->second, ", referred to by executor id ", ex_id));
  }
  ex_it->second.remote_refcount--;
  should_destroy = ex_it->second.remote_refcount == 0;
  if (should_destroy) {
    DestroyExecutorImpl(ex_id);
  }
  return absl::OkStatus();
}

void ExecutorService::ExecutorResolver::DestroyExecutor(const ExecutorId& id) {
  absl::MutexLock lock(&executors_mutex_);
  DestroyExecutorImpl(id);
}

void ExecutorService::ExecutorResolver::DestroyExecutorImpl(
    const ExecutorId& id) {
  VLOG(3) << "Destroying executor: " << id;
  auto ex_cardinalities = keys_to_cardinalities_.find(id);
  if (ex_cardinalities != keys_to_cardinalities_.end()) {
    executors_.erase(ex_cardinalities->second);
    keys_to_cardinalities_.erase(id);
  } else {
    VLOG(2) << "Attempted to double-destroy executor of key: " << id;
  }
}

grpc::Status ExecutorService::GetExecutor(grpc::ServerContext* context,
                                          const v0::GetExecutorRequest* request,
                                          v0::GetExecutorResponse* response) {
  CardinalityMap cardinalities;
  for (const auto& cardinality : request->cardinalities()) {
    cardinalities.insert(
        {cardinality.placement().uri(), cardinality.cardinality()});
  }
  absl::StatusOr<std::string> id_or =
      executor_resolver_.ExecutorIDForRequirements({cardinalities});
  if (!id_or.ok()) {
    return absl_to_grpc(id_or.status());
  }
  std::string id = id_or.value();
  *response->mutable_executor()->mutable_id() = id;
  return grpc::Status::OK;
}

grpc::Status ExecutorService::RequireExecutor(
    std::string_view method_name, const v0::ExecutorId& executor,
    std::shared_ptr<Executor>& executor_out) {
  absl::StatusOr<ExecutorEntry> ex =
      executor_resolver_.ExecutorForId({executor.id()});
  if (!ex.ok()) {
    absl::Status status_to_return(
        ex.status().code(), absl::StrCat("Error calling `", method_name, "`. ",
                                         ex.status().message()));
    return absl_to_grpc(status_to_return);
  }
  executor_out = ex.value().executor;
  return grpc::Status::OK;
}

grpc::Status ExecutorService::HandleNotOK(const absl::Status& status,
                                          const v0::ExecutorId& executor_id) {
  if (status.code() == absl::StatusCode::kFailedPrecondition) {
    // TODO(b/193900393): With increased reliance on the semantics of
    // FAILED_PRECONDITION, we would likely prefer to define a custom error code
    // for Executors to declare that they are missing semantically necessary
    // requirements.
    VLOG(1) << "Destroying executor " << executor_id.Utf8DebugString();
    executor_resolver_.DestroyExecutor({executor_id.id()});
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
  std::optional<ValueId> embedded_arg;
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
  grpc::Status executor_status =
      RequireExecutor("Dispose", request->executor(), executor);
  if (!executor_status.ok()) {
    LOG_FIRST_N(WARNING, 10) << "Received a dispose request for ["
                             << request->executor().id() << "], but it was not "
                             << "found.";
    // There may be no executor corresponding to this Dispose request, if the
    // underlying executor was destroyed before this request came in (e.g., in
    // the case of an executor returning FAILED_PRECONDITION). We consider the
    // Dispose request to have succeeded in this case; the value has certainly
    // been destroyed.
    return grpc::Status::OK;
  }
  std::vector<ValueId> embedded_ids_to_dispose;
  embedded_ids_to_dispose.reserve(request->value_ref().size());
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
  return absl_to_grpc(
      executor_resolver_.DisposeExecutor({request->executor().id()}));
}

}  // namespace tensorflow_federated

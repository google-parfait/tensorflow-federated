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

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
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

grpc::Status ParseRef(const v0::ValueRef& value_ref_pb, ValueId* value_id_out) {
  if (absl::SimpleAtoi(value_ref_pb.id(), value_id_out)) {
    return grpc::Status::OK;
  }
  return grpc::Status(
      grpc::StatusCode::INVALID_ARGUMENT,
      absl::StrCat("Could not parse ValueRef from string. Incoming id:",
                   value_ref_pb.id()));
}

grpc::Status ExecutorService::RequireExecutor_(
    std::string method_name, std::shared_ptr<Executor>* executor_out,
    int* generation_out) {
  absl::ReaderMutexLock reader_lock(&executor_mutex_);
  if (executor_and_generation_.first == nullptr) {
    return grpc::Status(
        grpc::StatusCode::UNAVAILABLE,
        absl::StrCat("Attempted to call ExecutorService::", method_name,
                     " before setting cardinalities."));
  }
  *executor_out = executor_and_generation_.first;
  *generation_out = executor_and_generation_.second;
  return grpc::Status::OK;
}

ExecutorService::RemoteValueId ExecutorService::CreateRemoteValue_(
    ValueId embedded_value_id, int executor_generation) {
  return absl::StrCat(embedded_value_id, "-", executor_generation);
}

grpc::Status MalformattedValueStatus(absl::string_view bad_id) {
  return grpc::Status(
      grpc::StatusCode::INVALID_ARGUMENT,
      absl::StrCat("Remote value ID ", bad_id,
                   " malformed: expected to be of the form a-b, where a and "
                   "b are both ints."));
}

grpc::Status ExecutorService::EnsureGeneration_(int reference_generation,
                                                int expected_generation) {
  if (reference_generation != expected_generation) {
    return grpc::Status(
        grpc::StatusCode::INVALID_ARGUMENT,
        absl::StrCat(
            "Remote value refers to a non-live executor generation. "
            "Current generation is: ",
            expected_generation,
            " remote value refers to generation: ", reference_generation));
  }
  return grpc::Status::OK;
}

grpc::Status ExecutorService::ResolveRemoteValue_(
    const v0::ValueRef& remote_value_ref, int expected_generation,
    ValueId* value_id_out) {
  // Incoming ref should have ID of the form a-b, where a is a
  // uint64_t and b s an int. a represents the ValueId in the
  // service's executor, b represents the generation of this executor.
  std::vector<absl::string_view> id_and_generation =
      absl::StrSplit(remote_value_ref.id(), '-');
  if (id_and_generation.size() != 2) {
    return MalformattedValueStatus(remote_value_ref.id());
  }
  int reference_generation;
  if (!absl::SimpleAtoi(id_and_generation[1], &reference_generation)) {
    return MalformattedValueStatus(remote_value_ref.id());
  }
  TFF_TRYLOG_GRPC(EnsureGeneration_(reference_generation, expected_generation));
  if (!absl::SimpleAtoi(id_and_generation[0], value_id_out)) {
    return MalformattedValueStatus(remote_value_ref.id());
  }
  return grpc::Status::OK;
}

grpc::Status ExecutorService::CreateValue(grpc::ServerContext* context,
                                          const v0::CreateValueRequest* request,
                                          v0::CreateValueResponse* response) {
  std::shared_ptr<Executor> executor;
  int generation;
  TFF_TRYLOG_GRPC(RequireExecutor_("CreateValue", &executor, &generation));
  absl::StatusOr<OwnedValueId> id = executor->CreateValue(request->value());
  if (!id.ok()) {
    LOG(ERROR) << id.status().message();
    return absl_to_grpc(id.status());
  }
  response->mutable_value_ref()->mutable_id()->assign(
      CreateRemoteValue_(id.value().ref(), generation));
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
  int generation;
  TFF_TRYLOG_GRPC(RequireExecutor_("CreateCall", &executor, &generation));
  ValueId embedded_fn;
  TFF_TRYLOG_GRPC(
      ResolveRemoteValue_(request->function_ref(), generation, &embedded_fn));
  absl::optional<ValueId> embedded_arg;
  if (request->has_argument_ref()) {
    // Callers should avoid setting the argument ref for invocation of a no-arg
    // fn.
    TFF_TRYLOG_GRPC(ResolveRemoteValue_(request->argument_ref(), generation,
                                        &embedded_arg.emplace()));
  }
  absl::StatusOr<OwnedValueId> called_fn =
      executor->CreateCall(embedded_fn, embedded_arg);
  if (!called_fn.ok()) {
    LOG(ERROR) << called_fn.status().message();
    return absl_to_grpc(called_fn.status());
  }
  response->mutable_value_ref()->mutable_id()->assign(
      CreateRemoteValue_(called_fn.value().ref(), generation));
  // We must prevent this destructor from running similarly to CreateValue.
  called_fn.value().forget();
  return grpc::Status::OK;
}

grpc::Status ExecutorService::CreateStruct(
    grpc::ServerContext* context, const v0::CreateStructRequest* request,
    v0::CreateStructResponse* response) {
  std::shared_ptr<Executor> executor;
  int generation;
  TFF_TRYLOG_GRPC(RequireExecutor_("CreateStruct", &executor, &generation));
  std::vector<ValueId> requested_ids;
  requested_ids.reserve(request->element().size());
  for (const v0::CreateStructRequest::Element& elem : request->element()) {
    ValueId id;
    TFF_TRYLOG_GRPC(ResolveRemoteValue_(elem.value_ref(), generation, &id));
    requested_ids.push_back(id);
  }
  absl::StatusOr<OwnedValueId> created_struct =
      executor->CreateStruct(requested_ids);
  if (!created_struct.ok()) {
    LOG(ERROR) << created_struct.status().message();
    return absl_to_grpc(created_struct.status());
  }
  response->mutable_value_ref()->mutable_id()->assign(
      CreateRemoteValue_(created_struct.value().ref(), generation));
  // We must prevent this destructor from running similarly to CreateValue.
  created_struct.value().forget();
  return grpc::Status::OK;
}

grpc::Status ExecutorService::CreateSelection(
    grpc::ServerContext* context, const v0::CreateSelectionRequest* request,
    v0::CreateSelectionResponse* response) {
  std::shared_ptr<Executor> executor;
  int generation;
  TFF_TRYLOG_GRPC(RequireExecutor_("CreateSelection", &executor, &generation));
  ValueId selection_source;
  TFF_TRYLOG_GRPC(ResolveRemoteValue_(request->source_ref(), generation,
                                      &selection_source));
  absl::StatusOr<OwnedValueId> selected_element =
      executor->CreateSelection(selection_source, request->index());
  if (!selected_element.ok()) {
    LOG(ERROR) << selected_element.status().message();
    return absl_to_grpc(selected_element.status());
  }
  response->mutable_value_ref()->mutable_id()->assign(
      CreateRemoteValue_(selected_element.value().ref(), generation));
  // We must prevent this destructor from running similarly to CreateValue.
  selected_element.value().forget();
  return grpc::Status::OK;
}

grpc::Status ExecutorService::Compute(grpc::ServerContext* context,
                                      const v0::ComputeRequest* request,
                                      v0::ComputeResponse* response) {
  std::shared_ptr<Executor> executor;
  int generation;
  TFF_TRYLOG_GRPC(RequireExecutor_("Compute", &executor, &generation));
  ValueId requested_value;
  TFF_TRYLOG_GRPC(
      ResolveRemoteValue_(request->value_ref(), generation, &requested_value));
  absl::Status status =
      executor->Materialize(requested_value, response->mutable_value());
  return absl_to_grpc(status);
}

grpc::Status ExecutorService::SetCardinalities(
    grpc::ServerContext* context, const v0::SetCardinalitiesRequest* request,
    v0::SetCardinalitiesResponse* response) {
  CardinalityMap cardinality_map;
  for (const auto& cardinality : request->cardinalities()) {
    cardinality_map.insert(
        {cardinality.placement().uri(), cardinality.cardinality()});
  }
  {
    absl::WriterMutexLock writer_lock(&executor_mutex_);
    absl::StatusOr<std::shared_ptr<Executor>> new_executor =
        executor_factory_(cardinality_map);
    if (!new_executor.ok()) {
      LOG(ERROR) << new_executor.status().message();
      return absl_to_grpc(new_executor.status());
    }
    int new_generation = executor_and_generation_.second + 1;
    executor_and_generation_ =
        std::make_pair(std::move(new_executor.value()), new_generation);
  }
  return grpc::Status::OK;
}

grpc::Status ExecutorService::ClearExecutor(
    grpc::ServerContext* context, const v0::ClearExecutorRequest* request,
    v0::ClearExecutorResponse* response) {
  {
    absl::WriterMutexLock writer_lock(&executor_mutex_);
    // Clearing executor should reset executor to null, but without changing the
    // executor's generation; this will be incremented by SetCardinalities.
    executor_and_generation_ =
        std::make_pair(nullptr, executor_and_generation_.second);
  }
  return grpc::Status::OK;
}

grpc::Status ExecutorService::Dispose(grpc::ServerContext* context,
                                      const v0::DisposeRequest* request,
                                      v0::DisposeResponse* response) {
  std::shared_ptr<Executor> executor;
  int generation;
  TFF_TRYLOG_GRPC(RequireExecutor_("Dispose", &executor, &generation));
  std::vector<ValueId> embedded_ids_to_dispose;
  embedded_ids_to_dispose.reserve(request->value_ref().size());
  // Filter the requested IDs to those corresponding to the currently live
  // executors. Clients are free to batch these requests however they want or
  // let them be called by a GC mechanism, so we do not force the client to only
  // dispose of values in the live executor.
  for (const v0::ValueRef& disposed_value_ref : request->value_ref()) {
    ValueId embedded_value;
    grpc::Status status =
        ResolveRemoteValue_(disposed_value_ref, generation, &embedded_value);
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

}  // namespace tensorflow_federated

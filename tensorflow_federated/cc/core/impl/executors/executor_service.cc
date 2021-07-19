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

#include <memory>
#include <tuple>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/proto/v0/executor.proto.h"

namespace tensorflow_federated {

inline absl::Status LogIfNotOk(absl::Status s) {
  if (!s.ok()) {
    LOG(ERROR) << s;
  }
  return s;
}

template <typename T>
inline absl::StatusOr<T> LogIfNotOk(absl::StatusOr<T> s) {
  if (!s.ok()) {
    LOG(ERROR) << s.status();
  }
  return s;
}

absl::StatusOr<ValueId> ParseRef(const v0::ValueRef& value_ref_pb) {
  ValueId value_id;
  if (absl::SimpleAtoi(value_ref_pb.id(), &value_id)) {
    return value_id;
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Could not parse ValueRef from string. Incoming id:", value_ref_pb.id()));
}

absl::StatusOr<std::pair<std::shared_ptr<Executor>, int>>
ExecutorService::RequireExecutor_(std::string method_name) {
  absl::ReaderMutexLock reader_lock(&executor_mutex_);
  if (executor_and_generation_.first == nullptr) {
    return grpc::Status(
        grpc::StatusCode::UNAVAILABLE,
        absl::StrCat("Attempted to call ExecutorService::", method_name,
                     " before setting cardinalities."));
  }
  return executor_and_generation_;
}

ExecutorService::RemoteValueId ExecutorService::CreateRemoteValue_(
    ValueId embedded_value_id, int executor_generation) {
  return absl::StrCat(embedded_value_id, "-", executor_generation);
}

absl::Status MalformattedValueStatus(absl::string_view bad_id) {
  return absl::InvalidArgumentError(
      absl::StrCat("Remote value ID ", bad_id,
                   " malformed: expected to be of the form a-b, where a and "
                   "b are both ints."));
}

absl::Status ExecutorService::EnsureGeneration_(int reference_generation,
                                                int expected_generation) {
  if (reference_generation != expected_generation) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Remote value refers to a non-live executor generation. "
        "Current generation is: ",
        expected_generation,
        "remote valuerefers to generation: ", reference_generation));
  }
  return absl::OkStatus();
}

absl::StatusOr<ValueId> ExecutorService::ResolveRemoteValue_(
    const v0::ValueRef& remote_value_ref, int expected_generation) {
  // Incoming ref should have ID of the form a-b, where a is a
  // uint64 and b s an int. a represents the ValueId in the
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
  TFF_TRY(
      LogIfNotOk(EnsureGeneration_(reference_generation, expected_generation)));

  ValueId embedded_value_id;
  if (!absl::SimpleAtoi(id_and_generation[0], &embedded_value_id)) {
    return MalformattedValueStatus(remote_value_ref.id());
  }
  return embedded_value_id;
}

grpc::Status ExecutorService::CreateValue(grpc::ServerContext* context,
                                          const v0::CreateValueRequest* request,
                                          v0::CreateValueResponse* response) {
  std::pair<std::shared_ptr<Executor>, int> executor_and_generation;
  auto [live_executor, used_executor_generation] =
      TFF_TRY(LogIfNotOk(RequireExecutor_("CreateValue")));
  OwnedValueId owned_value_id =
      TFF_TRY(LogIfNotOk(live_executor->CreateValue(request->value())));
  response->mutable_value_ref()->mutable_id()->assign(
      CreateRemoteValue_(owned_value_id.ref(), used_executor_generation));
  // We must call forget on the embedded id to prevent the destructor from
  // running when the variable goes out of scope. Similar considerations apply
  // to the reset of the Create methods below.
  owned_value_id.forget();
  return grpc::Status::OK;
}

grpc::Status ExecutorService::CreateCall(grpc::ServerContext* context,
                                         const v0::CreateCallRequest* request,
                                         v0::CreateCallResponse* response) {
  std::pair<std::shared_ptr<Executor>, int> executor_and_generation;
  auto [live_executor, used_executor_generation] =
      TFF_TRY(LogIfNotOk(RequireExecutor_("CreateCall")));
  ValueId embedded_fn = TFF_TRY(LogIfNotOk(
      ResolveRemoteValue_(request->function_ref(), used_executor_generation)));
  std::optional<ValueId> embedded_arg;
  if (request->has_argument_ref()) {
    // Callers should avoid setting the argument ref for invocation of a no-arg
    // fn.
    embedded_arg = TFF_TRY(LogIfNotOk(ResolveRemoteValue_(
        request->argument_ref(), used_executor_generation)));
  }
  OwnedValueId called_fn =
      TFF_TRY(LogIfNotOk(live_executor->CreateCall(embedded_fn, embedded_arg)));
  response->mutable_value_ref()->mutable_id()->assign(
      CreateRemoteValue_(called_fn.ref(), used_executor_generation));
  // We must prevent this destructor from running similarly to CreateValue.
  called_fn.forget();
  return grpc::Status::OK;
}

grpc::Status ExecutorService::CreateStruct(
    grpc::ServerContext* context, const v0::CreateStructRequest* request,
    v0::CreateStructResponse* response) {
  auto [live_executor, used_executor_generation] =
      TFF_TRY(LogIfNotOk(RequireExecutor_("CreateStruct")));
  std::vector<ValueId> requested_ids;
  requested_ids.reserve(request->element().size());
  for (const v0::CreateStructRequest::Element& elem : request->element()) {
    requested_ids.emplace_back(TFF_TRY(LogIfNotOk(
        ResolveRemoteValue_(elem.value_ref(), used_executor_generation))));
  }
  OwnedValueId created_struct =
      TFF_TRY(LogIfNotOk(live_executor->CreateStruct(requested_ids)));
  response->mutable_value_ref()->mutable_id()->assign(
      CreateRemoteValue_(created_struct.ref(), used_executor_generation));
  // We must prevent this destructor from running similarly to CreateValue.
  created_struct.forget();
  return grpc::Status::OK;
}

grpc::Status ExecutorService::CreateSelection(
    grpc::ServerContext* context, const v0::CreateSelectionRequest* request,
    v0::CreateSelectionResponse* response) {
  auto [live_executor, used_executor_generation] =
      TFF_TRY(LogIfNotOk(RequireExecutor_("CreateSelection")));
  ValueId selection_source = TFF_TRY(LogIfNotOk(
      ResolveRemoteValue_(request->source_ref(), used_executor_generation)));
  OwnedValueId selected_element = TFF_TRY(LogIfNotOk(
      live_executor->CreateSelection(selection_source, request->index())));
  response->mutable_value_ref()->mutable_id()->assign(
      CreateRemoteValue_(selected_element.ref(), used_executor_generation));
  // We must prevent this destructor from running similarly to CreateValue.
  selected_element.forget();
  return grpc::Status::OK;
}

grpc::Status ExecutorService::Compute(grpc::ServerContext* context,
                                      const v0::ComputeRequest* request,
                                      v0::ComputeResponse* response) {
  auto [live_executor, used_executor_generation] =
      TFF_TRY(LogIfNotOk(RequireExecutor_("Compute")));
  ValueId requested_value = TFF_TRY(LogIfNotOk(
      ResolveRemoteValue_(request->value_ref(), used_executor_generation)));
  TFF_TRY(LogIfNotOk(
      live_executor->Materialize(requested_value, response->mutable_value())));
  return grpc::Status::OK;
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
    auto new_executor = TFF_TRY(LogIfNotOk(executor_factory_(cardinality_map)));
    int new_generation = executor_and_generation_.second + 1;
    executor_and_generation_ =
        std::make_pair(std::move(new_executor), new_generation);
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
  auto [live_executor, used_executor_generation] =
      TFF_TRY(LogIfNotOk(RequireExecutor_("Dispose")));
  std::vector<ValueId> embedded_ids_to_dispose;
  embedded_ids_to_dispose.reserve(request->value_ref().size());
  // Filter the requested IDs to those corresponding to the currently live
  // executors. Clients are free to batch these requests however they want or
  // let them be called by a GC mechanism, so we do not force the client to only
  // dispose of values in the live executor.
  for (const v0::ValueRef& disposed_value_ref : request->value_ref()) {
    absl::StatusOr<ValueId> embedded_value =
        ResolveRemoteValue_(disposed_value_ref, used_executor_generation);
    if (embedded_value.ok()) {
      TFF_TRY(LogIfNotOk(live_executor->Dispose(embedded_value.ValueOrDie())));
    }
  }
  return grpc::Status::OK;
}

}  // namespace tensorflow_federated

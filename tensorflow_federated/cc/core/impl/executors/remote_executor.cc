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
#include "tensorflow_federated/cc/core/impl/executors/federated_intrinsics.h"
#include "tensorflow_federated/cc/core/impl/executors/status_conversion.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/threading.h"
#include "tensorflow_federated/cc/core/impl/executors/type_utils.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"
#include "tensorflow_federated/proto/v0/executor.grpc.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

class ExecutorValue;

using ValueFuture =
    std::shared_future<absl::StatusOr<std::shared_ptr<ExecutorValue>>>;

// Create a FederatedZip intrinsic.
//
// This method is only used when the RemoteExecutor is configured to use
// streaming mode. This method must set the appropriate type signature so that
// the remote executor can track the resulting value, which is necessary to
// later stream results during materialization.
absl::StatusOr<v0::Value> CreateFederatedZipComputation(
    const v0::StructType& parameter_type_pb,
    const v0::FederatedType& result_type_pb,
    const v0::PlacementSpec& placement_spec) {
  v0::Value intrinsic_pb;
  v0::Computation* computation_pb = intrinsic_pb.mutable_computation();
  v0::FunctionType* computation_type_pb =
      computation_pb->mutable_type()->mutable_function();
  *computation_type_pb->mutable_parameter()->mutable_struct_() =
      parameter_type_pb;
  *computation_type_pb->mutable_result()->mutable_federated() = result_type_pb;
  if (placement_spec.value().uri() == kClientsUri) {
    computation_pb->mutable_intrinsic()->mutable_uri()->assign(
        kFederatedZipAtClientsUri.data(), kFederatedZipAtClientsUri.length());
  } else if (placement_spec.value().uri() == kServerUri) {
    computation_pb->mutable_intrinsic()->mutable_uri()->assign(
        kFederatedZipAtServerUri.data(), kFederatedZipAtServerUri.length());
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Could not stream federated value with placement [",
                     placement_spec.ShortDebugString(),
                     "]. Intrinsic to zip after streaming not implemented"));
  }
  return intrinsic_pb;
}

// Creates a selection computation that is federated mapped to a federated
// value.
//
// This is necessary for streaming values, which can generate selections of the
// elements of the structure, but directly selecting from federated values is
// not supported.
v0::Value CreateSelectionComputation(const v0::StructType& parameter_type_pb,
                                     const v0::Type& result_type_pb,
                                     int32_t selection_index) {
  v0::Value value_pb;
  v0::FunctionType* lambda_type_pb =
      value_pb.mutable_computation()->mutable_type()->mutable_function();
  *lambda_type_pb->mutable_parameter()->mutable_struct_() = parameter_type_pb;
  *lambda_type_pb->mutable_result() = result_type_pb;

  v0::Lambda* lambda_pb = value_pb.mutable_computation()->mutable_lambda();
  lambda_pb->set_parameter_name("selection_arg");
  v0::Computation* selection_computation_pb = lambda_pb->mutable_result();
  v0::Selection* selection_pb = selection_computation_pb->mutable_selection();
  selection_pb->mutable_source()->mutable_reference()->set_name(
      "selection_arg");
  selection_pb->set_index(selection_index);
  return value_pb;
}

absl::StatusOr<v0::Value> CreateFederatedMapComputation(
    const v0::FederatedType& parameter_type_pb,
    const v0::FederatedType& result_type_pb) {
  v0::Value value_pb;
  v0::FunctionType* computation_type_pb =
      value_pb.mutable_computation()->mutable_type()->mutable_function();
  *computation_type_pb->mutable_parameter()->mutable_federated() =
      parameter_type_pb;
  *computation_type_pb->mutable_result()->mutable_federated() = result_type_pb;

  v0::Intrinsic* intrinsic_pb =
      value_pb.mutable_computation()->mutable_intrinsic();
  if (parameter_type_pb.placement().value().uri() == kClientsUri) {
    intrinsic_pb->mutable_uri()->assign(kFederatedMapAtClientsUri.data(),
                                        kFederatedMapAtClientsUri.length());
  } else if (parameter_type_pb.placement().value().uri() == kServerUri) {
    intrinsic_pb->mutable_uri()->assign(kFederatedMapAtServerUri.data(),
                                        kFederatedMapAtServerUri.length());
  } else {
    return absl::UnimplementedError(
        absl::StrCat("Cannot create federated map for placement: [",
                     parameter_type_pb.placement().value().uri(), "]"));
  }
  return value_pb;
}

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
                 // TODO(b/269766462): consider splitting into a second class
                 // rather than parameterizing on a boolean here.
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
  absl::Status MaterializeRPC(ValueFuture value, v0::Value* value_pb);
  absl::Status MaterializeStreaming(ValueFuture value, v0::Value* value_pb);

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

  absl::StatusOr<std::shared_ptr<ExecutorValue>> CreateValueRPC(
      const v0::Value& value_pb);
  absl::StatusOr<std::shared_ptr<ExecutorValue>> CreateExecutorValueStreaming(
      const v0::Value& value_pb);
  absl::StatusOr<std::shared_ptr<ExecutorValue>>
  CreateExecutorFederatedValueStreaming(const v0::Value& value_pb);
};

// A value tracked by the RemoteExecutor.
//
// Unlike other executors, the RemoteExecutor keeps track of some (but not all)
// type information for the values it operates on. This is required for
// operating in "streaming structures" mode, especially for materializing
// values from the executors on the other side of the StubInterface: if the
// value_ref refers to a structure (possibly federated, or containing federated
// values), the materialization must create selections for the elements of the
// structure and stream them back.
class ExecutorValue {
 public:
  ExecutorValue(v0::ValueRef value_ref, v0::Type type_pb,
                v0::ExecutorId executor_pb,
                std::shared_ptr<v0::ExecutorGroup::StubInterface> stub)
      : value_ref_(std::move(value_ref)),
        type_pb_(std::move(type_pb)),
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

  const v0::ValueRef& Get() const { return value_ref_; }
  const v0::Type& Type() const { return type_pb_; }

 private:
  const v0::ValueRef value_ref_;
  const v0::Type type_pb_;
  const v0::ExecutorId executor_pb_;
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

absl::StatusOr<std::shared_ptr<ExecutorValue>>
RemoteExecutor::CreateExecutorFederatedValueStreaming(
    const v0::Value& value_pb) {
  const v0::Value::Federated& federated_pb = value_pb.federated();
  if (!federated_pb.type().member().has_struct_()) {
    return CreateValueRPC(value_pb);
  }
  // If this is a federated struct, we need to convert the
  // "federated-structure-of-values" to "structure-of-federated-values"
  // for streaming across the RPC channel. At the end, a `federated_zip`
  // intrisic call will promote the values back to a
  // `federated_structure_of_values`.
  const v0::PlacementSpec& placement_spec = federated_pb.type().placement();
  const bool all_equal = federated_pb.type().all_equal();
  const int32_t struct_size =
      federated_pb.type().member().struct_().element_size();
  // We build up a type for the intrinsic parameter for the federated_zip
  // computation that will be called after the streaming structure.
  v0::StructType parameter_type_pb;
  std::vector<ValueFuture> elements;
  elements.reserve(struct_size);
  v0::Value element_pb;
  for (int32_t i = 0; i < struct_size; ++i) {
    element_pb.Clear();
    v0::Value::Federated* federated_element_pb = element_pb.mutable_federated();
    v0::FederatedType* federated_type = federated_element_pb->mutable_type();
    *federated_type->mutable_placement() = placement_spec;
    federated_type->set_all_equal(all_equal);
    // Note: ignoring the `name()` of the elements.
    *federated_type->mutable_member() =
        federated_pb.type().member().struct_().element(i).value();
    *parameter_type_pb.add_element()->mutable_value()->mutable_federated() =
        *federated_type;
    for (const v0::Value& child_value : federated_pb.value()) {
      *federated_element_pb->add_value() =
          child_value.struct_().element(i).value();
    }
    elements.push_back(TFF_TRY(CreateExecutorValue(element_pb)));
  }
  ValueFuture struct_value_ref = TFF_TRY(CreateStruct(std::move(elements)));
  // Now call a federated_zip intrinsics on the
  // structure-of-federated-values to promote it back to a
  // federated-structure-of-values.
  const v0::Value intrinsic_pb = TFF_TRY(CreateFederatedZipComputation(
      parameter_type_pb, federated_pb.type(), placement_spec));
  ValueFuture intrinsic_ref = TFF_TRY(CreateExecutorValue(intrinsic_pb));
  return TFF_TRY(Wait(TFF_TRY(CreateCall(intrinsic_ref, struct_value_ref))));
}

absl::StatusOr<std::shared_ptr<ExecutorValue>>
RemoteExecutor::CreateExecutorValueStreaming(const v0::Value& value_pb) {
  switch (value_pb.value_case()) {
    case v0::Value::kStruct: {
      const v0::Value::Struct& struct_pb = value_pb.struct_();
      std::vector<ValueFuture> elements;
      elements.reserve(struct_pb.element_size());
      for (const v0::Value::Struct::Element& element_pb : struct_pb.element()) {
        elements.push_back(TFF_TRY(CreateExecutorValue(element_pb.value())));
      }
      std::shared_ptr<ExecutorValue> value_ref =
          TFF_TRY(Wait(TFF_TRY(CreateStruct(std::move(elements)))));
      return std::move(value_ref);
    }
    case v0::Value::kFederated: {
      return CreateExecutorFederatedValueStreaming(value_pb);
    }
    default:
      // Simply forward any other types of values through, they will not be
      // streamed.
      return CreateValueRPC(value_pb);
  }
}

absl::StatusOr<ValueFuture> RemoteExecutor::CreateExecutorValue(
    const v0::Value& value_pb) {
  TFF_TRY(EnsureInitialized());
  if (stream_structs_) {
    return ThreadRun([value_pb, this, this_keepalive = shared_from_this()]()
                         -> absl::StatusOr<std::shared_ptr<ExecutorValue>> {
      return this->CreateExecutorValueStreaming(value_pb);
    });
  } else {
    return ThreadRun([value_pb, this, this_keepalive = shared_from_this()]()
                         -> absl::StatusOr<std::shared_ptr<ExecutorValue>> {
      return CreateValueRPC(value_pb);
    });
  }
}

absl::StatusOr<std::shared_ptr<ExecutorValue>> RemoteExecutor::CreateValueRPC(
    const v0::Value& value_pb) {
  v0::CreateValueRequest request;
  *request.mutable_executor() = executor_pb_;
  *request.mutable_value() = value_pb;
  v0::CreateValueResponse response;
  grpc::ClientContext client_context;
  grpc::Status status = stub_->CreateValue(&client_context, request, &response);
  TFF_TRY(grpc_to_absl(status));
  return std::make_shared<ExecutorValue>(std::move(response.value_ref()),
                                         TFF_TRY(InferTypeFromValue(value_pb)),
                                         executor_pb_, stub_);
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
    return std::make_shared<ExecutorValue>(std::move(response.value_ref()),
                                           fn->Type().function().result(),
                                           executor_pb, this->stub_);
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
    v0::Type result_type;
    v0::StructType* struct_type = result_type.mutable_struct_();
    for (const std::shared_ptr<ExecutorValue>& element : values) {
      v0::CreateStructRequest_Element struct_elem;
      *struct_elem.mutable_value_ref() = element->Get();
      *struct_type->add_element()->mutable_value() = element->Type();
      request.mutable_element()->Add(std::move(struct_elem));
    }
    grpc::Status status =
        this->stub_->CreateStruct(&context, request, &response);
    TFF_TRY(grpc_to_absl(status));
    auto result = std::make_shared<ExecutorValue>(
        std::move(response.value_ref()), std::move(result_type),
        this->executor_pb_, this->stub_);
    return result;
  });
}

absl::StatusOr<ValueFuture> RemoteExecutor::CreateSelection(
    ValueFuture value, const uint32_t index) {
  TFF_TRY(EnsureInitialized());
  return ThreadRun([source = std::move(value), index = index, this,
                    this_keepalive = shared_from_this()]()
                       -> absl::StatusOr<std::shared_ptr<ExecutorValue>> {
    std::shared_ptr<ExecutorValue> source_value = TFF_TRY(Wait(source));
    const v0::Type& source_type_pb = source_value->Type();
    if (source_type_pb.has_federated()) {
      // Note: normally a selection on a federated value would not be expected,
      // as this isn't possible to construct/trace (its not a language feature).
      // This path is encountered when streaming mode is enabled and a federated
      // value is being materialized. In such a case the remote executor needs
      // to JIT create computations to perform selections on the leaves of the
      // structure to stream them "out" of the executors.
      if (!source_type_pb.federated().member().has_struct_()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Error selecting from non-Struct value: ",
                         source_type_pb.ShortDebugString()));
      }
      // TODO(b/269766462): instead of creating a selection computation per
      // element, we could look at creating one lambda that includes the
      // federated_map+selection on all leaves at once. This would make even
      // more sense if splitting out to a `StreamingRemoteExecutor` class.
      const v0::Type element_type_pb =
          source_type_pb.federated().member().struct_().element(index).value();
      const v0::Value selection_comp = CreateSelectionComputation(
          source_type_pb.federated().member().struct_(), element_type_pb,
          index);
      ValueFuture selection_value =
          TFF_TRY(CreateExecutorValue(selection_comp));
      v0::FederatedType federated_element_type_pb;
      *federated_element_type_pb.mutable_placement() =
          source_type_pb.federated().placement();
      federated_element_type_pb.set_all_equal(
          source_type_pb.federated().all_equal());
      *federated_element_type_pb.mutable_member() = element_type_pb;
      const v0::Value federated_map_comp =
          TFF_TRY(CreateFederatedMapComputation(source_type_pb.federated(),
                                                federated_element_type_pb));
      ValueFuture map_value_ref =
          TFF_TRY(CreateExecutorValue(federated_map_comp));
      ValueFuture map_arg = TFF_TRY(CreateStruct({selection_value, source}));
      return TFF_TRY(Wait(TFF_TRY(CreateCall(map_value_ref, map_arg))));
    }
    if (!source_type_pb.has_struct_()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Error selecting from non-Struct value: ",
                       source_type_pb.ShortDebugString()));
    }
    v0::CreateSelectionRequest request;
    v0::CreateSelectionResponse response;
    grpc::ClientContext context;
    *request.mutable_executor() = this->executor_pb_;
    *request.mutable_source_ref() = source_value->Get();
    request.set_index(index);
    grpc::Status status =
        this->stub_->CreateSelection(&context, request, &response);
    const v0::Type element_type_pb =
        source_type_pb.struct_().element(index).value();
    TFF_TRY(grpc_to_absl(status));
    return std::make_shared<ExecutorValue>(std::move(response.value_ref()),
                                           std::move(element_type_pb),
                                           this->executor_pb_, this->stub_);
  });
}

absl::Status RemoteExecutor::MaterializeStreaming(ValueFuture value,
                                                  v0::Value* value_pb) {
  std::shared_ptr<ExecutorValue> value_ref = TFF_TRY(Wait(value));
  switch (value_ref->Type().type_case()) {
    case v0::Type::kTensor: {
      return MaterializeRPC(value, value_pb);
    }
    case v0::Type::kStruct: {
      const v0::StructType& struct_type_pb = value_ref->Type().struct_();
      v0::Value::Struct* struct_value_pb = value_pb->mutable_struct_();
      for (int32_t i = 0; i < struct_type_pb.element_size(); ++i) {
        ValueFuture selection = TFF_TRY(CreateSelection(value, i));
        TFF_TRY(Materialize(selection,
                            struct_value_pb->add_element()->mutable_value()));
      }
      return absl::OkStatus();
    }
    case v0::Type::kFederated: {
      const v0::Type& member_type_pb = value_ref->Type().federated().member();
      if (!member_type_pb.has_struct_()) {
        // If not struct, nothing to stream; forward call as-is.
        return MaterializeRPC(value, value_pb);
      }
      v0::Value::Federated* federated_pb = value_pb->mutable_federated();
      *federated_pb->mutable_type() = value_ref->Type().federated();
      // Otherwise we need to stream the federated structure by creating
      // a selection for each element and materializing them individually which
      // creates a struct-of-federated-values.
      const v0::StructType& struct_type_pb = member_type_pb.struct_();
      // If this is a federated empty tuple, simply return that.
      if (struct_type_pb.element_size() == 0) {
        return absl::OkStatus();
      }
      v0::Value::Struct struct_value_pb;
      for (int32_t i = 0; i < struct_type_pb.element_size(); ++i) {
        ValueFuture selection = TFF_TRY(CreateSelection(value, i));
        TFF_TRY(Materialize(selection,
                            struct_value_pb.add_element()->mutable_value()));
      }
      // Now convert the struct-of-federated-values back to a
      // federated-struct-of-values. We've already returned above if this is
      // an empty federated structure, so grab the cardinality off the first
      // structure element.
      const int32_t placement_cardinality =
          struct_value_pb.element(0).value().federated().value_size();
      for (int32_t placement_index = 0; placement_index < placement_cardinality;
           ++placement_index) {
        v0::Value::Struct* federated_struct_pb =
            federated_pb->add_value()->mutable_struct_();
        for (int32_t struct_index = 0;
             struct_index < struct_value_pb.element_size(); ++struct_index) {
          v0::Value::Struct::Element* element_pb =
              federated_struct_pb->add_element();
          if (struct_value_pb.element(struct_index)
                  .value()
                  .federated()
                  .value_size() <= placement_index) {
            return absl::InvalidArgumentError(absl::StrCat(
                "Expect cardinality [", placement_cardinality, "] but got [",
                struct_value_pb.element(struct_index)
                    .value()
                    .federated()
                    .value_size(),
                "]: ", struct_value_pb.ShortDebugString()));
          }
          element_pb->mutable_value()->Swap(
              struct_value_pb.mutable_element(struct_index)
                  ->mutable_value()
                  ->mutable_federated()
                  ->mutable_value(placement_index));
        }
      }
      return absl::OkStatus();
    }
    default:
      break;  // Forward all other types through.
  }
  return MaterializeRPC(value, value_pb);
}

absl::Status RemoteExecutor::Materialize(ValueFuture value,
                                         v0::Value* value_pb) {
  if (stream_structs_) {
    return MaterializeStreaming(value, value_pb);
  } else {
    return MaterializeRPC(value, value_pb);
  }
}

absl::Status RemoteExecutor::MaterializeRPC(ValueFuture value,
                                            v0::Value* value_pb) {
  std::shared_ptr<ExecutorValue> value_ref = TFF_TRY(Wait(value));
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

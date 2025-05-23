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

#include "tensorflow_federated/cc/core/impl/executors/streaming_remote_executor.h"

#include <climits>
#include <cstdint>
#include <future>  // NOLINT
#include <list>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "include/grpcpp/grpcpp.h"
#include "include/grpcpp/support/status.h"
#include "federated_language/proto/computation.pb.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/federated_intrinsics.h"
#include "tensorflow_federated/cc/core/impl/executors/status_conversion.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/threading.h"
#include "tensorflow_federated/cc/core/impl/executors/type_utils.h"
#include "tensorflow_federated/proto/v0/executor.grpc.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

class ExecutorValue;

using ValueFuture =
    std::shared_future<absl::StatusOr<std::shared_ptr<ExecutorValue>>>;

// Create a structure by extracting all the values inside the federated values
// of a structure.
//
// This method is used repeatedly to turn a structure-of-federated-values into
// a federated-structure-of-values.
absl::Status BuildPlacedStructValue(const v0::Value::Struct& struct_value_pb,
                                    int32_t placement_index,
                                    v0::Value::Struct* placed_struct_pb) {
  for (const v0::Value::Struct::Element& element_pb :
       struct_value_pb.element()) {
    v0::Value::Struct::Element* placed_element_pb =
        placed_struct_pb->add_element();
    switch (element_pb.value().value_case()) {
      case v0::Value::kFederated: {
        if (element_pb.value().federated().value_size() <= placement_index) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Expect cardinality [", placement_index, "] but got [",
              element_pb.value().federated().value_size(),
              "]: ", struct_value_pb.ShortDebugString()));
        }
        *placed_element_pb->mutable_value() =
            element_pb.value().federated().value(placement_index);
        break;
      }
      case v0::Value::kStruct: {
        TFF_TRY(BuildPlacedStructValue(
            element_pb.value().struct_(), placement_index,
            placed_element_pb->mutable_value()->mutable_struct_()));
        break;
      }
      default:
        return absl::InternalError(
            "Found non-federated element when trying to create a placed value "
            "from a structure of federated values");
    }
  }
  return absl::OkStatus();
}

// Create a FederatedZip intrinsic.
//
// This method is only used when the RemoteExecutor is configured to use
// streaming mode. This method must set the appropriate type signature so that
// the remote executor can track the resulting value, which is necessary to
// later stream results during materialization.
absl::StatusOr<v0::Value> CreateFederatedZipComputation(
    const federated_language::StructType& parameter_type_pb,
    const federated_language::FederatedType& result_type_pb,
    const federated_language::PlacementSpec& placement_spec) {
  v0::Value intrinsic_pb;
  federated_language::Computation* computation_pb =
      intrinsic_pb.mutable_computation();
  federated_language::FunctionType* computation_type_pb =
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

federated_language::Call CreateCalledFederatedMappedSelection(
    absl::string_view intrinsic_uri, absl::string_view arg_ref_name,
    int32_t index) {
  federated_language::Call call_pb;
  call_pb.mutable_function()->mutable_intrinsic()->set_uri(
      intrinsic_uri.data(), intrinsic_uri.size());
  federated_language::Struct* arg_struct =
      call_pb.mutable_argument()->mutable_struct_();

  federated_language::Lambda* local_lambda_pb =
      arg_struct->add_element()->mutable_value()->mutable_lambda();
  constexpr char kMapArg[] = "map_arg";
  local_lambda_pb->set_parameter_name(kMapArg);
  federated_language::Selection* selection_pb =
      local_lambda_pb->mutable_result()->mutable_selection();
  selection_pb->mutable_source()->mutable_reference()->set_name(kMapArg);
  selection_pb->set_index(index);

  arg_struct->add_element()->mutable_value()->mutable_reference()->set_name(
      arg_ref_name.data(), arg_ref_name.size());
  return call_pb;
}

// Creates a selection computation that is federated mapped to a federated
// value, selecting all the leaves.
//
// Effectively, this method turns a `federated-structure-of-values` into a
// `structure-of-federated-values`.  This is necessary for streaming values,
// which can generate selections of the elements of the structure, but directly
// selecting from federated values is not supported.
//
// The computation created is a federated_map that applies a multi-selection
// computation to a federated structure structure. In TFF computation shorthand,
// a simple example of this would look like:
//
//   (map_arg -> (let
//      local_0=federated_map(<
//        (select_arg -> select_arg[0]),
//        map_arg
//      >),
//      local_1=federated_map(<
//        (select_arg -> select_arg[1]),
//        map_arg
//      >)
//     in <local_0, local_1>
//   ))
absl::StatusOr<v0::Value> CreateSelectionFederatedStructComputation(
    const federated_language::FederatedType& parameter_type_pb) {
  if (!parameter_type_pb.member().has_struct_()) {
    // We don't want to create and send RPCs for computations that don't require
    // them, make this an error condition.
    return absl::InvalidArgumentError(
        absl::StrCat("parameter_type_pb is a not a federated structure type: ",
                     parameter_type_pb.ShortDebugString()));
  }
  v0::Value value_pb;
  federated_language::FunctionType* lambda_type_pb =
      value_pb.mutable_computation()->mutable_type()->mutable_function();
  *lambda_type_pb->mutable_parameter()->mutable_federated() = parameter_type_pb;
  // NOTE: the result type will be computed as we build the computation and set
  // at the end of this method.
  federated_language::StructType result_type_pb;
  federated_language::FederatedType federated_type_template_pb;
  *federated_type_template_pb.mutable_placement() =
      parameter_type_pb.placement();
  federated_type_template_pb.set_all_equal(parameter_type_pb.all_equal());

  federated_language::Lambda* lambda_pb =
      value_pb.mutable_computation()->mutable_lambda();
  constexpr char kFederatedStructArg[] = "federated_struct_arg";
  lambda_pb->set_parameter_name(kFederatedStructArg);

  absl::string_view intrinsic_uri;
  if (parameter_type_pb.placement().value().uri() == kClientsUri) {
    intrinsic_uri = kFederatedMapAtClientsUri;
  } else if (parameter_type_pb.placement().value().uri() == kServerUri) {
    intrinsic_uri = kFederatedMapAtServerUri;
  } else {
    return absl::UnimplementedError(
        absl::StrCat("Cannot create federated map for placement: [",
                     parameter_type_pb.placement().value().uri(), "]"));
  }

  // A list of elements to iteratively process as the method descends into a
  // nested structure. We perform a breadth-first-traversal of the nested
  // structure.
  std::list<std::tuple<federated_language::Block*, std::string,
                       const federated_language::StructType*,
                       federated_language::StructType*>>
      structs_to_process = {{
          lambda_pb->mutable_result()->mutable_block(),
          kFederatedStructArg,
          &parameter_type_pb.member().struct_(),
          &result_type_pb,
      }};

  while (!structs_to_process.empty()) {
    auto [block_pb, parent_ref_name, parent_struct_type_pb,
          output_struct_type_pb] = structs_to_process.front();
    structs_to_process.pop_front();
    for (int32_t i = 0; i < parent_struct_type_pb->element_size(); ++i) {
      const federated_language::StructType::Element& element_type_pb =
          parent_struct_type_pb->element(i);
      federated_language::StructType::Element* output_element_type_pb =
          output_struct_type_pb->add_element();
      switch (element_type_pb.value().type_case()) {
        case federated_language::Type::kTensor:
        case federated_language::Type::kSequence: {
          federated_language::Block::Local* local_pb = block_pb->add_local();
          local_pb->set_name(absl::StrCat("elem_", i));
          *local_pb->mutable_value()->mutable_call() =
              CreateCalledFederatedMappedSelection(intrinsic_uri,
                                                   parent_ref_name, i);
          *output_element_type_pb->mutable_value()->mutable_federated() =
              federated_type_template_pb;
          *output_element_type_pb->mutable_value()
               ->mutable_federated()
               ->mutable_member() = element_type_pb.value();
          break;
        }
        case federated_language::Type::kStruct: {
          // Add a local to select the nested structure, and give it a name with
          // the selection path.
          std::string nested_struct_ref_name =
              absl::StrCat("nested_struct_", i);
          federated_language::Block::Local* nested_struct_local_pb =
              block_pb->add_local();
          nested_struct_local_pb->set_name(nested_struct_ref_name);
          *nested_struct_local_pb->mutable_value()->mutable_call() =
              CreateCalledFederatedMappedSelection(intrinsic_uri,
                                                   parent_ref_name, i);
          // We now need to descend into this struct, add it to the list to
          // process.
          federated_language::Block::Local* nested_block_pb =
              block_pb->add_local();
          nested_block_pb->set_name(absl::StrCat("elem_", i));
          structs_to_process.emplace_back(
              nested_block_pb->mutable_value()->mutable_block(),
              nested_struct_ref_name, &element_type_pb.value().struct_(),
              output_element_type_pb->mutable_value()->mutable_struct_());
          break;
        }
        default:
          return absl::UnimplementedError(
              absl::StrCat("Cannot handle structs containing [",
                           element_type_pb.value().type_case(), "] types"));
      }
    }
    // After traversing all the elements in the current structure, gather the
    // elements from the locals that need to be part of the output structure.
    federated_language::Struct* result_struct =
        block_pb->mutable_result()->mutable_struct_();
    for (const federated_language::Block::Local& local_pb : block_pb->local()) {
      // Only pickup the elements, not local selections, in the file output.
      if (absl::StartsWith(local_pb.name(), "elem_")) {
        result_struct->add_element()
            ->mutable_value()
            ->mutable_reference()
            ->set_name(local_pb.name());
      }
    }
  }
  // Set the result type now that it has been computed.
  *lambda_type_pb->mutable_result()->mutable_struct_() = result_type_pb;
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

class StreamingRemoteExecutor : public ExecutorBase<ValueFuture> {
 public:
  StreamingRemoteExecutor(
      std::unique_ptr<v0::ExecutorGroup::StubInterface> stub,
      const CardinalityMap& cardinalities)
      : stub_(stub.release(), StubDeleter()), cardinalities_(cardinalities) {}

  ~StreamingRemoteExecutor() override = default;

  absl::string_view ExecutorName() final {
    static constexpr absl::string_view kExecutorName =
        "StreamingRemoteExecutor";
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

 private:
  absl::Status EnsureInitialized();
  std::shared_ptr<v0::ExecutorGroup::StubInterface> stub_;
  CardinalityMap cardinalities_;
  absl::Mutex mutex_;
  bool executor_pb_set_ ABSL_GUARDED_BY(mutex_) = false;
  v0::ExecutorId executor_pb_;

  absl::StatusOr<ValueFuture> CreateValueRPC(const v0::Value& value_pb);
  absl::StatusOr<ValueFuture> CreateExecutorValueStreaming(
      const v0::Value& value_pb);
  absl::StatusOr<ValueFuture> CreateExecutorFederatedValueStreaming(
      const v0::Value& value_pb);
};

// A value tracked by the StreamingRemoteExecutor.
//
// Unlike other executors, the StreamingRemoteExecutor keeps track of some (but
// not all) type information for the values it operates on. This is required for
// operating in "streaming structures" mode, especially for materializing
// values from the executors on the other side of the StubInterface: if the
// value_ref refers to a structure (possibly federated, or containing federated
// values), the materialization must create selections for the elements of the
// structure and stream them back.
class ExecutorValue {
 public:
  ExecutorValue(v0::ValueRef value_ref, federated_language::Type type_pb,
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
  const federated_language::Type& Type() const { return type_pb_; }

 private:
  const v0::ValueRef value_ref_;
  const federated_language::Type type_pb_;
  const v0::ExecutorId executor_pb_;
  std::shared_ptr<v0::ExecutorGroup::StubInterface> stub_;
};

absl::Status StreamingRemoteExecutor::EnsureInitialized() {
  absl::MutexLock lock(&mutex_);
  if (executor_pb_set_) {
    return absl::OkStatus();
  }
  v0::GetExecutorRequest request;
  for (auto iter = cardinalities_.begin(); iter != cardinalities_.end();
       ++iter) {
    federated_language::Placement placement;
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

absl::StatusOr<ValueFuture>
StreamingRemoteExecutor::CreateExecutorFederatedValueStreaming(
    const v0::Value& value_pb) {
  const v0::Value::Federated& federated_pb = value_pb.federated();
  if (!federated_pb.type().member().has_struct_()) {
    return CreateValueRPC(value_pb);
  }
  // If this is a federated struct, we need to convert the
  // "federated-structure-of-values" to "structure-of-federated-values" for
  // streaming across the RPC channel. At the end, a `federated_zip` intrisic
  // call will promote the values back to a `federated_structure_of_values`.
  const federated_language::PlacementSpec& placement_spec =
      federated_pb.type().placement();
  const bool all_equal = federated_pb.type().all_equal();
  const int32_t struct_size =
      federated_pb.type().member().struct_().element_size();
  if (struct_size == 0) {
    // Empty struct has nothing to stream, send the value as-is.
    return CreateValueRPC(value_pb);
  }
  // We build up a type for the intrinsic parameter for the federated_zip
  // computation that will be called after the streaming structure.
  federated_language::StructType parameter_type_pb;
  std::vector<ValueFuture> elements;
  elements.reserve(struct_size);
  v0::Value element_pb;
  for (int32_t i = 0; i < struct_size; ++i) {
    element_pb.Clear();
    v0::Value::Federated* federated_element_pb = element_pb.mutable_federated();
    federated_language::FederatedType* federated_type =
        federated_element_pb->mutable_type();
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
  // Now call a federated_zip intrinsics on the structure-of-federated-values to
  // promote it back to a federated-structure-of-values.
  const v0::Value intrinsic_pb = TFF_TRY(CreateFederatedZipComputation(
      parameter_type_pb, federated_pb.type(), placement_spec));
  ValueFuture intrinsic_ref = TFF_TRY(CreateExecutorValue(intrinsic_pb));
  return CreateCall(intrinsic_ref, struct_value_ref);
}

absl::StatusOr<ValueFuture>
StreamingRemoteExecutor::CreateExecutorValueStreaming(
    const v0::Value& value_pb) {
  switch (value_pb.value_case()) {
    case v0::Value::kStruct: {
      const v0::Value::Struct& struct_pb = value_pb.struct_();
      std::vector<ValueFuture> elements;
      elements.reserve(struct_pb.element_size());
      for (const v0::Value::Struct::Element& element_pb : struct_pb.element()) {
        elements.push_back(
            TFF_TRY(CreateExecutorValueStreaming(element_pb.value())));
      }
      return CreateStruct(std::move(elements));
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

absl::StatusOr<ValueFuture> StreamingRemoteExecutor::CreateExecutorValue(
    const v0::Value& value_pb) {
  TFF_TRY(EnsureInitialized());
  return ThreadRun([value_pb, this, this_keepalive = shared_from_this()]()
                       -> absl::StatusOr<std::shared_ptr<ExecutorValue>> {
    return TFF_TRY(Wait(TFF_TRY(this->CreateExecutorValueStreaming(value_pb))));
  });
}

absl::StatusOr<ValueFuture> StreamingRemoteExecutor::CreateValueRPC(
    const v0::Value& value_pb) {
  federated_language::Type type_pb = TFF_TRY(InferTypeFromValue(value_pb));
  VLOG(5) << "CreateValueRPC: [" << type_pb.ShortDebugString() << "]";
  if (type_pb.has_function() || type_pb.ShortDebugString().empty()) {
    VLOG(5) << value_pb.Utf8DebugString();
  }
  if (value_pb.ByteSizeLong() > INT_MAX) {
    if (type_pb.has_federated()) {
      LOG(ERROR) << "Federated type `" << type_pb.ShortDebugString()
                 << "` with " << value_pb.federated().value_size()
                 << " values.";
    }
    return absl::InvalidArgumentError(absl::StrCat(
        "Message with type `", type_pb.ShortDebugString(),
        "` will fail to serialize for gRPC, size: ", value_pb.ByteSizeLong()));
  }
  v0::CreateValueRequest request;
  *request.mutable_executor() = executor_pb_;
  *request.mutable_value() = value_pb;
  v0::CreateValueResponse response;
  grpc::ClientContext client_context;
  grpc::Status status = stub_->CreateValue(&client_context, request, &response);
  TFF_TRY(grpc_to_absl(status));
  return ReadyFuture(
      std::make_shared<ExecutorValue>(std::move(response.value_ref()),
                                      std::move(type_pb), executor_pb_, stub_));
}

absl::StatusOr<ValueFuture> StreamingRemoteExecutor::CreateCall(
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

absl::StatusOr<ValueFuture> StreamingRemoteExecutor::CreateStruct(
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
    federated_language::Type result_type;
    federated_language::StructType* struct_type = result_type.mutable_struct_();
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

absl::StatusOr<ValueFuture> StreamingRemoteExecutor::CreateSelection(
    ValueFuture value, const uint32_t index) {
  TFF_TRY(EnsureInitialized());
  return ThreadRun([source = std::move(value), index = index, this,
                    this_keepalive = shared_from_this()]()
                       -> absl::StatusOr<std::shared_ptr<ExecutorValue>> {
    std::shared_ptr<ExecutorValue> source_value = TFF_TRY(Wait(source));
    const federated_language::Type& source_type_pb = source_value->Type();
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
    const federated_language::Type element_type_pb =
        source_type_pb.struct_().element(index).value();
    TFF_TRY(grpc_to_absl(status));
    return std::make_shared<ExecutorValue>(std::move(response.value_ref()),
                                           std::move(element_type_pb),
                                           this->executor_pb_, this->stub_);
  });
}

absl::Status StreamingRemoteExecutor::Materialize(ValueFuture value,
                                                  v0::Value* value_pb) {
  std::shared_ptr<ExecutorValue> value_ref = TFF_TRY(Wait(value));
  switch (value_ref->Type().type_case()) {
    case federated_language::Type::kTensor: {
      return MaterializeRPC(value, value_pb);
    }
    case federated_language::Type::kStruct: {
      const federated_language::StructType& struct_type_pb =
          value_ref->Type().struct_();
      v0::Value::Struct* struct_value_pb = value_pb->mutable_struct_();
      for (int32_t i = 0; i < struct_type_pb.element_size(); ++i) {
        ValueFuture selection = TFF_TRY(CreateSelection(value, i));
        TFF_TRY(Materialize(selection,
                            struct_value_pb->add_element()->mutable_value()));
      }
      return absl::OkStatus();
    }
    case federated_language::Type::kFederated: {
      const federated_language::Type& member_type_pb =
          value_ref->Type().federated().member();
      if (!member_type_pb.has_struct_()) {
        // If not struct, nothing to stream; forward call as-is.
        return MaterializeRPC(value, value_pb);
      }
      int32_t cardinality = 0;
      if (value_ref->Type().federated().placement().value().uri() ==
          kServerUri) {
        cardinality = 1;
      } else {
        auto iter = cardinalities_.find(
            value_ref->Type().federated().placement().value().uri());
        if (iter == cardinalities_.end()) {
          return absl::InternalError(absl::StrCat(
              "Somehow tried to materialize a value with placed [",
              value_ref->Type().federated().placement().value().uri(),
              "] which is unknown. Only have cardinalites for: ",
              absl::StrJoin(cardinalities_, ",", absl::PairFormatter("="))));
        }
        cardinality = iter->second;
      }
      if (cardinality <= 0) {
        return absl::InternalError(absl::StrCat(
            "Computed cardinality for placement ",
            value_ref->Type().federated().placement().value().uri(), " is ",
            cardinality, " <= 0!. Had cardinality map: ",
            absl::StrJoin(cardinalities_, ",", absl::PairFormatter("=")),
            ". This indicates an internal coding/logic error in the "
            "system."));
      }
      if (member_type_pb.struct_().element_size() == 0) {
        // Empty struct has nothing to materialize from the remote, avoid
        // issuing the RPC and simply return.
        v0::Value::Federated* federated_pb = value_pb->mutable_federated();
        for (int32_t i = 0; i < cardinality; ++i) {
          federated_pb->add_value()->mutable_struct_();
        }
        *federated_pb->mutable_type() = value_ref->Type().federated();
        return absl::OkStatus();
      }
      // Otherwise we need to stream the federated structure by creating
      // a selection for each element and materializing them individually which
      // creates a struct-of-federated-values.
      ValueFuture selection_computation = TFF_TRY(
          CreateExecutorValue(TFF_TRY(CreateSelectionFederatedStructComputation(
              value_ref->Type().federated()))));
      v0::Value intermediate_value_pb;
      TFF_TRY(Materialize(TFF_TRY(CreateCall(selection_computation, value)),
                          &intermediate_value_pb));
      if (intermediate_value_pb.struct_().element_size() == 0) {
        return absl::InternalError(absl::StrCat(
            "Something went wrong, got empty struct but expect value for type ",
            value_ref->Type().ShortDebugString()));
      }
      // We need to convert from the materialized struct-of-federated-values
      // back to a federated-struct-of-values.
      const v0::Value::Struct struct_value_pb = intermediate_value_pb.struct_();
      v0::Value::Federated* federated_pb = value_pb->mutable_federated();
      for (int32_t placement_index = 0; placement_index < cardinality;
           ++placement_index) {
        v0::Value::Struct* federated_struct_pb =
            federated_pb->add_value()->mutable_struct_();
        TFF_TRY(BuildPlacedStructValue(struct_value_pb, placement_index,
                                       federated_struct_pb));
      }
      *federated_pb->mutable_type() = value_ref->Type().federated();
      return absl::OkStatus();
    }
    default:
      break;  // Forward all other types through.
  }
  return MaterializeRPC(value, value_pb);
}

absl::Status StreamingRemoteExecutor::MaterializeRPC(ValueFuture value,
                                                     v0::Value* value_pb) {
  std::shared_ptr<ExecutorValue> value_ref = TFF_TRY(Wait(value));
  VLOG(5) << "MaterializeRPC (" << value_ref->Get().ShortDebugString() << "): ["
          << value_ref->Type().ShortDebugString() << "]";
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

std::shared_ptr<Executor> CreateStreamingRemoteExecutor(
    std::unique_ptr<v0::ExecutorGroup::StubInterface> stub,
    const CardinalityMap& cardinalities) {
  return std::make_shared<StreamingRemoteExecutor>(std::move(stub),
                                                   cardinalities);
}

std::shared_ptr<Executor> CreateStreamingRemoteExecutor(
    std::shared_ptr<grpc::ChannelInterface> channel,
    const CardinalityMap& cardinalities) {
  std::unique_ptr<v0::ExecutorGroup::StubInterface> stub(
      v0::ExecutorGroup::NewStub(channel));
  return std::make_shared<StreamingRemoteExecutor>(std::move(stub),
                                                   cardinalities);
}
}  // namespace tensorflow_federated

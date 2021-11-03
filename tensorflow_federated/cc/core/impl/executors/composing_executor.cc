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

#include "tensorflow_federated/cc/core/impl/executors/composing_executor.h"

#include <cstddef>
#include <functional>
#include <future>  // NOLINT
#include <string>
#include <tuple>

#include "google/protobuf/repeated_field.h"
#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
#include "tensorflow_federated/cc/core/impl/executors/computations.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/federated_intrinsics.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/threading.h"
#include "tensorflow_federated/cc/core/impl/executors/value_validation.h"
#include "tensorflow_federated/proto/v0/computation.pb.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

namespace {

class ExecutorValue;

using ValueFuture = std::shared_future<absl::StatusOr<ExecutorValue>>;
using Children = std::tuple<uint32_t>;

inline std::shared_ptr<OwnedValueId> ShareValueId(OwnedValueId&& id) {
  return std::make_shared<OwnedValueId>(std::move(id));
}

// Inner (behind shared_ptr) representation of an unplaced value.
//
// The primary purpose of this class is to manage values which may be either:
// (1) a v0::Value proto
// (2) embedded in the `server` executor or
// (3) both.
class UnplacedInner {
 public:
  UnplacedInner(std::shared_ptr<v0::Value> proto) : proto_(std::move(proto)) {}
  UnplacedInner(v0::Value proto)
      : proto_(std::make_shared<v0::Value>(std::move(proto))) {}
  UnplacedInner(std::shared_ptr<OwnedValueId> embedded)
      : embedded_(std::move(embedded)) {}
  UnplacedInner(OwnedValueId embedded)
      : embedded_(std::make_shared<OwnedValueId>(std::move(embedded))) {}
  // NOTE: all constructors must set either `proto` OR (inclusive) `embedded`.
  // Internals assume that at least one of these values is set.

  // Returns the underlying `Proto` value without attempting to create one via
  // a `Materialize` call.
  absl::optional<absl::StatusOr<std::shared_ptr<v0::Value>>> GetProto() {
    absl::ReaderMutexLock lock(&mutex_);
    return proto_;
  }

  // Gets the proto representation of the inner value. May block on a remote
  // `Materialize` call.
  absl::StatusOr<std::shared_ptr<v0::Value>> Proto(Executor& server) {
    {
      // Try to grab the proto if it already exists.
      absl::ReaderMutexLock lock(&mutex_);
      if (proto_.has_value()) {
        return ExtractProto_();
      }
    }
    {
      // Try to grab the proto if it has been created since the last lock was
      // released.
      absl::WriterMutexLock lock(&mutex_);
      if (proto_.has_value()) {
        return ExtractProto_();
      }
      // Materialize the value from the underlying executor into a proto.
      auto proto_or_status =
          server.Materialize(embedded_.value().value()->ref());
      if (proto_or_status.ok()) {
        proto_ =
            std::make_shared<v0::Value>(std::move(proto_or_status.value()));
      } else {
        proto_ = std::move(proto_or_status.status());
      }
      return ExtractProto_();
    }
  }

  // Gets the embedded representation of the inner value.
  // May trigger a remote `CreateValue` call.
  absl::StatusOr<std::shared_ptr<OwnedValueId>> Embedded(Executor& server) {
    {
      // Try to grab the embedded ID if it already exists.
      absl::ReaderMutexLock lock(&mutex_);
      if (embedded_.has_value()) {
        return ExtractEmbedded_();
      }
    }
    {
      // Try to grab the embedded ID if it has been created since the last lock
      // was released.
      absl::WriterMutexLock lock(&mutex_);
      if (embedded_.has_value()) {
        return ExtractEmbedded_();
      }
      // Embed the proto into the underlying executor to get an ID.
      auto id_or_status = server.CreateValue(*proto_.value().value());
      if (id_or_status.ok()) {
        embedded_ = ShareValueId(std::move(id_or_status.value()));
      } else {
        embedded_ = std::move(id_or_status.status());
      }
      return ExtractEmbedded_();
    }
  }

 private:
  absl::StatusOr<std::shared_ptr<v0::Value>> ExtractProto_() const
      ABSL_SHARED_LOCKS_REQUIRED(mutex_) {
    if (proto_.value().ok()) {
      return proto_.value().value();
    } else {
      return proto_->status();
    }
  }

  absl::StatusOr<std::shared_ptr<OwnedValueId>> ExtractEmbedded_() const
      ABSL_SHARED_LOCKS_REQUIRED(mutex_) {
    if (embedded_.value().ok()) {
      return embedded_.value().value();
    } else {
      return embedded_->status();
    }
  }

  absl::Mutex mutex_;
  absl::optional<absl::StatusOr<std::shared_ptr<v0::Value>>> proto_
      ABSL_GUARDED_BY(mutex_);
  absl::optional<absl::StatusOr<std::shared_ptr<OwnedValueId>>> embedded_
      ABSL_GUARDED_BY(mutex_);
};

// NOTE: all variants must remain small and cheaply copyable (possibly via
// shared_ptr wrappers) so that `ExecutorValue` can be cheaply copied.
using Unplaced = std::shared_ptr<UnplacedInner>;
using Server = std::shared_ptr<OwnedValueId>;
using Clients = std::shared_ptr<std::vector<std::shared_ptr<OwnedValueId>>>;
using Structure = std::shared_ptr<std::vector<ExecutorValue>>;
using ValueVariant = absl::variant<Unplaced, Server, Clients, Structure,
                                   enum FederatedIntrinsic>;

inline Clients NewClients(uint32_t num_clients) {
  auto v = std::make_shared<std::vector<std::shared_ptr<OwnedValueId>>>();
  v->reserve(num_clients);
  return v;
}

inline Structure NewStructure() {
  return std::make_shared<std::vector<ExecutorValue>>();
}

class ExecutorValue {
 public:
  enum class ValueType { UNPLACED, SERVER, CLIENTS, STRUCTURE, INTRINSIC };

  inline const Unplaced& unplaced() const {
    return absl::get<::tensorflow_federated::Unplaced>(value_);
  }
  inline static ExecutorValue CreateUnplaced(
      ::tensorflow_federated::Unplaced id) {
    return ExecutorValue(std::move(id), ValueType::UNPLACED);
  }
  inline const Server& server() const {
    return absl::get<::tensorflow_federated::Server>(value_);
  }
  inline static ExecutorValue CreateServerPlaced(Server id) {
    return ExecutorValue(std::move(id), ValueType::SERVER);
  }
  inline const Clients& clients() const {
    return absl::get<::tensorflow_federated::Clients>(value_);
  }
  inline static ExecutorValue CreateClientsPlaced(Clients client_values) {
    return ExecutorValue(std::move(client_values), ValueType::CLIENTS);
  }
  // Convenience constructor from an un-shared_ptr vector.
  inline static ExecutorValue CreateClientsPlaced(
      std::vector<std::shared_ptr<OwnedValueId>>&& client_values) {
    return CreateClientsPlaced(
        std::make_shared<std::vector<std::shared_ptr<OwnedValueId>>>(
            std::move(client_values)));
  }
  inline const Structure& structure() const {
    return absl::get<::tensorflow_federated::Structure>(value_);
  }
  inline static ExecutorValue CreateStructure(Structure elements) {
    return ExecutorValue(std::move(elements), ValueType::STRUCTURE);
  }
  inline static ExecutorValue FederatedIntrinsic(FederatedIntrinsic intrinsic) {
    return ExecutorValue(intrinsic, ValueType::INTRINSIC);
  }
  inline enum FederatedIntrinsic intrinsic() const {
    return absl::get<enum FederatedIntrinsic>(value_);
  }

  absl::Status CheckLenForUseAsArgument(absl::string_view function_name,
                                        size_t len) const {
    if (type() != ExecutorValue::ValueType::STRUCTURE) {
      return absl::InvalidArgumentError(absl::StrCat(
          function_name,
          " expected a structural argument, found argument of type ", type()));
    }
    if (structure()->size() != len) {
      return absl::InvalidArgumentError(
          absl::StrCat(function_name, " expected ", len, " arguments, found ",
                       structure()->size()));
    }
    return absl::OkStatus();
  }

  // Fetches an unplaced functional value as a proto, ensuring that no
  // underlying `Materialize` call occurs.
  absl::StatusOr<std::shared_ptr<v0::Value>> GetUnplacedFunctionProto(
      absl::string_view name) const {
    if (type() != ExecutorValue::ValueType::UNPLACED) {
      return absl::InvalidArgumentError(
          absl::StrCat("`", name, "` must be an unplaced functional value, ",
                       "found a value of type ", type()));
    }
    auto opt_proto = unplaced()->GetProto();
    if (!opt_proto.has_value()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Cannot use a computed embedded value as unplaced function `", name,
          "`."));
    }
    return opt_proto.value();
  }

  absl::StatusOr<std::shared_ptr<OwnedValueId>> Embed(
      Executor& unplaced_child) const {
    switch (type()) {
      case ExecutorValue::ValueType::CLIENTS:
      case ExecutorValue::ValueType::SERVER: {
        return absl::InvalidArgumentError("Cannot embed a federated value.");
      }
      case ExecutorValue::ValueType::UNPLACED: {
        return unplaced()->Embedded(unplaced_child);
      }
      case ExecutorValue::ValueType::INTRINSIC: {
        return absl::InvalidArgumentError(
            "Cannot embed a federated intrinsic.");
      }
      case ExecutorValue::ValueType::STRUCTURE: {
        std::vector<std::shared_ptr<OwnedValueId>> field_ids;
        std::vector<ValueId> unowned_field_ids;
        size_t num_fields = structure()->size();
        field_ids.reserve(num_fields);
        unowned_field_ids.reserve(num_fields);
        for (const auto& field_value : *structure()) {
          auto child_field_value_id =
              TFF_TRY(field_value.Embed(unplaced_child));
          unowned_field_ids.push_back(child_field_value_id->ref());
          field_ids.push_back(std::move(child_field_value_id));
        }
        return ShareValueId(
            TFF_TRY(unplaced_child.CreateStruct(unowned_field_ids)));
      }
    }
  }

  // Constructs an `ExecutorValue` from a provided proto, delegating creation
  // of federated values to `create_federated` and creating unrecognized
  // values inside of the `unplaced_child` executor.
  static absl::StatusOr<ExecutorValue> FromProto(
      const v0::Value& value_pb, Executor& unplaced_child, uint32_t num_clients,
      const std::function<absl::StatusOr<ExecutorValue>(
          FederatedKind, const v0::Value_Federated&)>& create_federated) {
    switch (value_pb.value_case()) {
      case v0::Value::kFederated: {
        const v0::Value_Federated& federated = value_pb.federated();
        auto kind = TFF_TRY(ValidateFederated(num_clients, federated));
        return create_federated(kind, federated);
      }
      case v0::Value::kStruct: {
        auto elements = NewStructure();
        elements->reserve(value_pb.struct_().element_size());
        for (const auto& element_pb : value_pb.struct_().element()) {
          elements->emplace_back(TFF_TRY(
              ExecutorValue::FromProto(element_pb.value(), unplaced_child,
                                       num_clients, create_federated)));
        }
        return ExecutorValue::CreateStructure(std::move(elements));
      }
      case v0::Value::kComputation: {
        if (value_pb.computation().has_intrinsic()) {
          return ExecutorValue::FederatedIntrinsic(
              TFF_TRY(FederatedIntrinsicFromUri(
                  value_pb.computation().intrinsic().uri())));
        }
      }
        TF_FALLTHROUGH_INTENDED;
      default: {
        return ExecutorValue::CreateUnplaced(
            std::make_shared<UnplacedInner>(value_pb));
      }
    }
  }

  inline ValueType type() const { return type_; }

  ExecutorValue(ValueVariant value, ValueType type)
      : value_(std::move(value)), type_(type) {}

 private:
  ExecutorValue() = delete;
  ValueVariant value_;
  ValueType type_;
};

class ComposingExecutor : public ExecutorBase<ValueFuture> {
 public:
  explicit ComposingExecutor(std::shared_ptr<Executor> server,
                             std::vector<ComposingChild> children,
                             uint32_t total_clients)
      : server_(std::move(server)),
        children_(std::move(children)),
        total_clients_(total_clients) {}
  ~ComposingExecutor() override {
    // Delete `OwnedValueId` and release them from the child executor before
    // destroying it.
    ClearTracked();
  }

 protected:
  const char* ExecutorName() final { return "ComposingExecutor"; }

  absl::StatusOr<ValueFuture> CreateExecutorValue(
      const v0::Value& value_pb) final {
    return ReadyFuture(TFF_TRY(ExecutorValue::FromProto(
        value_pb, *server_, total_clients_, [this](auto kind, const auto& v) {
          return CreateFederatedValue_(kind, v);
        })));
  }

  absl::StatusOr<ValueFuture> CreateCall(
      ValueFuture function, absl::optional<ValueFuture> argument) final {
    return ThreadRun(
        [function = std::move(function), argument = std::move(argument), this,
         this_keepalive =
             shared_from_this()]() -> absl::StatusOr<ExecutorValue> {
          ExecutorValue fn = TFF_TRY(Wait(function));
          absl::optional<ExecutorValue> arg = absl::nullopt;
          if (argument.has_value()) {
            arg = TFF_TRY(Wait(argument.value()));
          }

          switch (fn.type()) {
            case ExecutorValue::ValueType::CLIENTS:
            case ExecutorValue::ValueType::SERVER: {
              return absl::InvalidArgumentError(
                  "Cannot call a federated value.");
            }
            case ExecutorValue::ValueType::STRUCTURE: {
              return absl::InvalidArgumentError("Cannot call a structure.");
            }
            case ExecutorValue::ValueType::UNPLACED: {
              // We need to materialize functions into the server
              // executor in order to execute them.
              auto fn_id = TFF_TRY(fn.unplaced()->Embedded(*server_));
              absl::optional<std::shared_ptr<OwnedValueId>> arg_owner;
              absl::optional<ValueId> arg_id = absl::nullopt;
              if (arg.has_value()) {
                arg_owner = TFF_TRY(arg.value().Embed(*server_));
                arg_id = arg_owner.value()->ref();
              }
              return ExecutorValue::CreateUnplaced(
                  std::make_shared<UnplacedInner>(
                      TFF_TRY(server_->CreateCall(fn_id->ref(), arg_id))));
            }
            case ExecutorValue::ValueType::INTRINSIC: {
              if (!arg.has_value()) {
                return absl::InvalidArgumentError(
                    "no argument provided for federated intrinsic");
              }
              return this->CallFederatedIntrinsic_(fn.intrinsic(),
                                                   std::move(arg.value()));
            }
          }
        });
  }

  absl::StatusOr<ValueFuture> CreateStruct(
      std::vector<ValueFuture> members) final {
    return Map(
        std::move(members),
        [](std::vector<ExecutorValue>&& members)
            -> absl::StatusOr<ExecutorValue> {
          return ExecutorValue::CreateStructure(
              std::make_shared<std::vector<ExecutorValue>>(std::move(members)));
        });
  }

  absl::StatusOr<ValueFuture> CreateSelection(ValueFuture value,
                                              const uint32_t index) final {
    return Map(
        std::vector<ValueFuture>({value}),
        [server = this->server_, index](std::vector<ExecutorValue>&& values)
            -> absl::StatusOr<ExecutorValue> {
          ExecutorValue& value = values[0];
          switch (value.type()) {
            case ExecutorValue::ValueType::CLIENTS:
            case ExecutorValue::ValueType::SERVER: {
              return absl::InvalidArgumentError(
                  "Cannot select from federated value");
            }
            case ExecutorValue::ValueType::UNPLACED: {
              auto id = TFF_TRY(value.unplaced()->Embedded(*server));
              return ExecutorValue::CreateUnplaced(
                  std::make_shared<UnplacedInner>(
                      TFF_TRY(server->CreateSelection(id->ref(), index))));
            }
            case ExecutorValue::ValueType::INTRINSIC: {
              return absl::InvalidArgumentError("Cannot select from intrinsic");
            }
            case ExecutorValue::ValueType::STRUCTURE: {
              if (value.structure()->size() <= index) {
                return absl::InvalidArgumentError(absl::StrCat(
                    "Invalid selection of index ", index,
                    " from structure of length ", value.structure()->size()));
              }
              return value.structure()->at(index);
            }
          }
        });
  }

  absl::Status Materialize(ValueFuture value_fut, v0::Value* value_pb) final {
    ExecutorValue value = TFF_TRY(Wait(std::move(value_fut)));
    ParallelTasks tasks;
    TFF_TRY(MaterializeValue_(value, value_pb, tasks));
    TFF_TRY(tasks.WaitAll());
    return absl::OkStatus();
  }

 private:
  std::shared_ptr<Executor> server_;
  std::vector<ComposingChild> children_;
  uint32_t total_clients_;

  Clients NewClients() const {
    return ::tensorflow_federated::NewClients(children_.size());
  }

  absl::StatusOr<ExecutorValue> CreateFederatedValue_(
      FederatedKind kind, const v0::Value_Federated& federated) {
    switch (kind) {
      case FederatedKind::SERVER: {
        auto value = TFF_TRY(server_->CreateValue(federated.value(0)));
        return ExecutorValue::CreateServerPlaced(
            ShareValueId(std::move(value)));
      }
      case FederatedKind::CLIENTS: {
        auto clients = NewClients();
        uint32_t next_client_index = 0;
        for (uint32_t i = 0; i < children_.size(); i++) {
          auto child = children_[i];
          v0::Value child_value;
          v0::Value_Federated* child_value_fed =
              child_value.mutable_federated();
          *child_value_fed->mutable_type() = federated.type();
          uint32_t stop_index = next_client_index + child.num_clients();
          for (; next_client_index < stop_index; next_client_index++) {
            *child_value_fed->add_value() = federated.value(next_client_index);
          }
          auto child_id = TFF_TRY(child.executor()->CreateValue(child_value));
          clients->emplace_back(ShareValueId(std::move(child_id)));
        }
        return ExecutorValue::CreateClientsPlaced(std::move(clients));
      }
      case FederatedKind::CLIENTS_ALL_EQUAL: {
        v0::Value child_value;
        *child_value.mutable_federated() = federated;
        return AllEqualToAll_(child_value);
      }
    }
  }

  v0::Value NewAllEqual_() const {
    v0::Value value;
    v0::Value_Federated* fed_value = value.mutable_federated();
    fed_value->mutable_type()->set_all_equal(true);
    fed_value->mutable_type()
        ->mutable_placement()
        ->mutable_value()
        ->mutable_uri()
        ->assign(kClientsUri.data(), kClientsUri.size());
    return value;
  }

  absl::StatusOr<ExecutorValue> AllEqualToAll_(
      const v0::Value& all_equal_value) const {
    auto clients = NewClients();
    for (const auto& child : children_) {
      auto child_id = TFF_TRY(child.executor()->CreateValue(all_equal_value));
      clients->emplace_back(ShareValueId(std::move(child_id)));
    }
    return ExecutorValue::CreateClientsPlaced(std::move(clients));
  }

  absl::StatusOr<ExecutorValue> CallIntrinsicValueAtClients_(
      ExecutorValue&& arg) const {
    if (arg.type() != ExecutorValue::ValueType::UNPLACED) {
      return absl::InvalidArgumentError(
          absl::StrCat("Cannot call federated_value_at_clients on value that "
                       "is not unplaced. Received "
                       "value of type: ",
                       arg.type()));
    }
    v0::Value value = NewAllEqual_();
    v0::Value* value_contents = value.mutable_federated()->add_value();
    *value_contents = *TFF_TRY(arg.unplaced()->Proto(*server_));
    return AllEqualToAll_(value);
  }

  absl::StatusOr<ExecutorValue> CallIntrinsicValueAtServer_(
      ExecutorValue&& arg) const {
    return ExecutorValue::CreateServerPlaced(TFF_TRY(arg.Embed(*server_)));
  }

  absl::StatusOr<ExecutorValue> CallIntrinsicEvalAtServer_(
      ExecutorValue&& arg) const {
    auto embedded = TFF_TRY(arg.Embed(*server_));
    return ExecutorValue::CreateServerPlaced(ShareValueId(
        TFF_TRY(server_->CreateCall(embedded->ref(), absl::nullopt))));
  }

  absl::StatusOr<ExecutorValue> CallIntrinsicEvalAtClients_(
      ExecutorValue&& arg) const {
    auto fn_to_eval =
        TFF_TRY(arg.GetUnplacedFunctionProto("federated_eval_at_clients_fn"));
    auto clients = NewClients();
    for (const auto& child : children_) {
      v0::Value eval_at_clients;
      eval_at_clients.mutable_computation()
          ->mutable_intrinsic()
          ->mutable_uri()
          ->assign(kFederatedEvalAtClientsUri.data(),
                   kFederatedEvalAtClientsUri.size());
      auto eval_id = TFF_TRY(child.executor()->CreateValue(eval_at_clients));
      auto fn_id = TFF_TRY(child.executor()->CreateValue(*fn_to_eval));
      auto res_id = TFF_TRY(child.executor()->CreateCall(eval_id, fn_id));
      clients->emplace_back(ShareValueId(std::move(res_id)));
    }
    return ExecutorValue::CreateClientsPlaced(std::move(clients));
  }

  absl::StatusOr<ExecutorValue> CallIntrinsicAggregate_(
      ExecutorValue&& arg) const {
    TFF_TRY(arg.CheckLenForUseAsArgument("federated_aggregate", 5));
    const auto& value = arg.structure()->at(0);
    if (value.type() != ExecutorValue::ValueType::CLIENTS) {
      return absl::InvalidArgumentError(
          "Cannot aggregate a value not placed at clients");
    }
    const auto& zero = arg.structure()->at(1);
    v0::Value zero_val;
    ParallelTasks tasks;
    TFF_TRY(MaterializeValue_(zero, &zero_val, tasks));
    TFF_TRY(tasks.WaitAll());

    const auto& accumulate = arg.structure()->at(2);
    auto accumulate_val =
        TFF_TRY(accumulate.GetUnplacedFunctionProto("accumulate"));

    const auto& merge = arg.structure()->at(3);
    auto merge_val = TFF_TRY(merge.GetUnplacedFunctionProto("merge"));
    auto merge_id = TFF_TRY(merge.unplaced()->Embedded(*server_));

    const auto& report = arg.structure()->at(4);
    auto report_val = TFF_TRY(report.GetUnplacedFunctionProto("report"));
    auto report_id = TFF_TRY(report.unplaced()->Embedded(*server_));

    v0::Value null_report_val;
    *null_report_val.mutable_computation() = IdentityComp();
    v0::Value aggregate;
    aggregate.mutable_computation()->mutable_intrinsic()->mutable_uri()->assign(
        kFederatedAggregateUri.data(), kFederatedAggregateUri.size());

    // Initiate the aggregation in each child.
    std::vector<OwnedValueId> child_result_ids;
    child_result_ids.reserve(children_.size());
    for (uint32_t i = 0; i < children_.size(); i++) {
      const auto& child = children_[i].executor();
      ValueId child_val = value.clients()->at(i)->ref();
      std::vector<OwnedValueId> arg_owners;
      std::vector<ValueId> arg_ids;
      arg_ids.emplace_back(child_val);
      for (const v0::Value& arg_value :
           {zero_val, *accumulate_val, *merge_val, null_report_val}) {
        OwnedValueId child_id = TFF_TRY(child->CreateValue(arg_value));
        arg_ids.emplace_back(child_id.ref());
        arg_owners.emplace_back(std::move(child_id));
      }
      auto child_arg_id = TFF_TRY(child->CreateStruct(std::move(arg_ids)));
      auto child_aggregate_id = TFF_TRY(child->CreateValue(aggregate));
      auto child_result_id =
          TFF_TRY(child->CreateCall(child_aggregate_id, child_arg_id));
      child_result_ids.push_back(std::move(child_result_id));
    }

    // Materialize and merge the results from each child executor.
    // TODO(b/192457028): parallelize this so that we're materializing more than
    // one value at a time and so that we can begin merging as soon as any
    // result is available.
    absl::optional<OwnedValueId> current = absl::nullopt;
    for (uint32_t i = 0; i < children_.size(); i++) {
      const auto& child = children_[i].executor();
      v0::Value child_result = TFF_TRY(child->Materialize(child_result_ids[i]));
      if (!child_result.has_federated() ||
          child_result.federated().type().placement().value().uri() !=
              kServerUri) {
        return absl::InternalError(
            "Child executor returned non-server-placed value");
      }
      auto child_result_server_id =
          TFF_TRY(server_->CreateValue(child_result.federated().value(0)));
      if (current.has_value()) {
        auto merge_arg = TFF_TRY(
            server_->CreateStruct({current.value(), child_result_server_id}));
        current = TFF_TRY(server_->CreateCall(merge_id->ref(), merge_arg));
      } else {
        current = std::move(child_result_server_id);
      }
    }
    auto result =
        TFF_TRY(server_->CreateCall(report_id->ref(), current.value()));
    return ExecutorValue::CreateServerPlaced(ShareValueId(std::move(result)));
  }

  absl::StatusOr<ExecutorValue> CallIntrinsicBroadcast_(
      ExecutorValue&& arg) const {
    if (arg.type() != ExecutorValue::ValueType::SERVER) {
      return absl::InvalidArgumentError(
          "Attempted to broadcast a value not placed at server.");
    }
    v0::Value value = NewAllEqual_();
    v0::Value* value_contents = value.mutable_federated()->add_value();
    TFF_TRY(server_->Materialize(arg.server()->ref(), value_contents));
    return AllEqualToAll_(value);
  }

  absl::StatusOr<ExecutorValue> CallIntrinsicMap_(ExecutorValue&& arg) const {
    TFF_TRY(arg.CheckLenForUseAsArgument("federated_map", 2));
    const auto& fn = arg.structure()->at(0);
    const auto& data = arg.structure()->at(1);
    if (data.type() == ExecutorValue::ValueType::CLIENTS) {
      Clients results = NewClients();
      v0::Value fn_val;
      ParallelTasks tasks;
      TFF_TRY(MaterializeValue_(fn, &fn_val, tasks));
      TFF_TRY(tasks.WaitAll());
      v0::Value map_val;
      map_val.mutable_computation()->mutable_intrinsic()->mutable_uri()->assign(
          kFederatedMapAtClientsUri.data(), kFederatedMapAtClientsUri.size());
      for (uint32_t i = 0; i < children_.size(); i++) {
        const auto& child = children_[i].executor();
        auto child_map = TFF_TRY(child->CreateValue(map_val));
        auto child_fn = TFF_TRY(child->CreateValue(fn_val));
        auto child_data = data.clients()->at(i)->ref();
        auto map_args = TFF_TRY(child->CreateStruct({child_fn, child_data}));
        auto result = TFF_TRY(child->CreateCall(child_map, map_args));
        results->emplace_back(ShareValueId(std::move(result)));
      }
      return ExecutorValue::CreateClientsPlaced(std::move(results));
    } else if (data.type() == ExecutorValue::ValueType::SERVER) {
      auto embedded_fn = TFF_TRY(fn.Embed(*server_));
      auto res = TFF_TRY(
          server_->CreateCall(embedded_fn->ref(), data.server()->ref()));
      return ExecutorValue::CreateServerPlaced(ShareValueId(std::move(res)));
    } else {
      return absl::InvalidArgumentError(
          "Attempted to map non-federated value.");
    }
  }

  absl::StatusOr<ExecutorValue> CallIntrinsicZip_(ExecutorValue&& arg) const {
    TFF_TRY(arg.CheckLenForUseAsArgument("federated_zip", 2));
    const auto& first = arg.structure()->at(0);
    const auto& second = arg.structure()->at(1);
    if (first.type() != second.type()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Attempted to zip values with different placements: ", first.type(),
          " and ", second.type()));
    }
    if (first.type() == ExecutorValue::ValueType::CLIENTS) {
      v0::Value zip_at_clients;
      zip_at_clients.mutable_computation()
          ->mutable_intrinsic()
          ->mutable_uri()
          ->assign(kFederatedZipAtClientsUri.data(),
                   kFederatedZipAtClientsUri.size());
      auto pairs = NewClients();
      for (uint32_t i = 0; i < children_.size(); i++) {
        auto child = children_[i].executor();
        auto zip = TFF_TRY(child->CreateValue(zip_at_clients));
        auto arg = TFF_TRY(child->CreateStruct({
            first.clients()->at(i)->ref(),
            second.clients()->at(i)->ref(),
        }));
        auto res = TFF_TRY(child->CreateCall(zip, arg));
        pairs->emplace_back(ShareValueId(std::move(res)));
      }
      return ExecutorValue::CreateClientsPlaced(std::move(pairs));
    } else if (first.type() == ExecutorValue::ValueType::SERVER) {
      ValueId first_id = first.server()->ref();
      ValueId second_id = second.server()->ref();
      auto pair = TFF_TRY(server_->CreateStruct({first_id, second_id}));
      return ExecutorValue::CreateServerPlaced(ShareValueId(std::move(pair)));
    } else {
      return absl::InvalidArgumentError(absl::StrCat(
          "Attempted to zip non-federated value of type ", first.type()));
    }
  }

  absl::StatusOr<ExecutorValue> CallFederatedIntrinsic_(
      FederatedIntrinsic function, ExecutorValue arg) const {
    switch (function) {
      case FederatedIntrinsic::VALUE_AT_CLIENTS: {
        return CallIntrinsicValueAtClients_(std::move(arg));
      }
      case FederatedIntrinsic::VALUE_AT_SERVER: {
        return CallIntrinsicValueAtServer_(std::move(arg));
      }
      case FederatedIntrinsic::EVAL_AT_SERVER: {
        return CallIntrinsicEvalAtServer_(std::move(arg));
      }
      case FederatedIntrinsic::EVAL_AT_CLIENTS: {
        return CallIntrinsicEvalAtClients_(std::move(arg));
      }
      case FederatedIntrinsic::AGGREGATE: {
        return CallIntrinsicAggregate_(std::move(arg));
      }
      case FederatedIntrinsic::BROADCAST: {
        return CallIntrinsicBroadcast_(std::move(arg));
      }
      case FederatedIntrinsic::MAP: {
        return CallIntrinsicMap_(std::move(arg));
      }
      case FederatedIntrinsic::ZIP: {
        return CallIntrinsicZip_(std::move(arg));
      }
    }
  }

  // Creates tasks to materialize values into the addresses pointed to by
  // `protos_out`.
  void MaterializeChildClientValues(uint32_t child_index, ValueId child_id,
                                    absl::Span<v0::Value*> protos_out,
                                    ParallelTasks& tasks) const {
    CHECK(protos_out.size() == children_[child_index].num_clients());
    tasks.add_task([child = children_[child_index], child_id,
                    protos_out]() -> absl::Status {
      v0::Value child_value = TFF_TRY(child.executor()->Materialize(child_id));
      if (!child_value.has_federated()) {
        return absl::InternalError(
            absl::StrCat("Composing child executor returned non-federated "
                         "value of type ",
                         child_value.value_case()));
      }
      if (child_value.federated().type().all_equal()) {
        if (child_value.federated().value_size() != 1) {
          return absl::InternalError(absl::StrCat(
              "Composing child executor returned all-equal value of "
              "length ",
              child_value.federated().value_size(),
              ", but all-equal values must have only one value."));
        }
        for (uint32_t j = 0; j < child.num_clients(); j++) {
          *(protos_out[j]) = child_value.federated().value(0);
        }
      } else {
        if (child_value.federated().value_size() != child.num_clients()) {
          return absl::InternalError(absl::StrCat(
              "Composing child executor responsible for ", child.num_clients(),
              " clients returned ", child_value.federated().value_size(),
              " client values."));
        }
        for (uint32_t j = 0; j < child.num_clients(); j++) {
          *(protos_out[j]) =
              std::move(*child_value.mutable_federated()->mutable_value(j));
        }
      }
      return absl::OkStatus();
    });
  }

  absl::Status MaterializeValue_(const ExecutorValue& value,
                                 v0::Value* value_pb,
                                 ParallelTasks& tasks) const {
    switch (value.type()) {
      case ExecutorValue::ValueType::CLIENTS: {
        v0::Value_Federated* federated_pb = value_pb->mutable_federated();
        google::protobuf::RepeatedPtrField<v0::Value>* values_pb =
            federated_pb->mutable_value();
        // Ensure that the `values_pb` array does not grow, changing the
        // addresses to which `MaterializeChildClientValues` should load.
        values_pb->Reserve(total_clients_);
        // Create `v0::Value`s for `MaterializeChildClientValues` to write to.
        // Note: we'd like to use `AddNAlreadyReserved`, but unfortunately that
        // method only exists for `RepeatedField`, not `RepeatedPtrField`.
        v0::Value null_value;
        for (uint32_t i = 0; i < total_clients_; i++) {
          *values_pb->Add() = null_value;
        }
        v0::FederatedType* type_pb = federated_pb->mutable_type();
        // All-equal-ness is not stored, so must be assumed to be false.
        // If the Python type system expects the value to be all-equal, it can
        // simply extract the first element in the list.
        type_pb->set_all_equal(false);
        type_pb->mutable_placement()->mutable_value()->mutable_uri()->assign(
            kClientsUri.data(), kClientsUri.size());
        v0::Value** client_start = values_pb->mutable_data();
        for (uint32_t i = 0; i < children_.size(); i++) {
          absl::Span<v0::Value*> client_value_pointers(
              client_start, children_[i].num_clients());
          ValueId child_value_id = value.clients()->at(i)->ref();
          MaterializeChildClientValues(i, child_value_id, client_value_pointers,
                                       tasks);
          client_start += children_[i].num_clients();
        }
        return absl::OkStatus();
      }
      case ExecutorValue::ValueType::UNPLACED: {
        tasks.add_task(
            [value = std::move(value), value_pb, server = server_]() {
              *value_pb = *TFF_TRY(value.unplaced()->Proto(*server));
              return absl::OkStatus();
            });
        return absl::OkStatus();
      }
      case ExecutorValue::ValueType::INTRINSIC: {
        return absl::UnimplementedError(
            "Materialization of federated intrinsics is not supported.");
      }
      case ExecutorValue::ValueType::SERVER: {
        v0::Value_Federated* federated_pb = value_pb->mutable_federated();
        v0::FederatedType* type_pb = federated_pb->mutable_type();
        // Server placement is assumed to be of cardinality one, and so must
        // be all-equal.
        type_pb->set_all_equal(true);
        type_pb->mutable_placement()->mutable_value()->mutable_uri()->assign(
            kServerUri.data(), kServerUri.size());
        tasks.add_task([value = std::move(value),
                        ptr = federated_pb->add_value(), server = server_]() {
          return server->Materialize(value.server()->ref(), ptr);
        });
        return absl::OkStatus();
      }
      case ExecutorValue::ValueType::STRUCTURE: {
        v0::Value_Struct* struct_pb = value_pb->mutable_struct_();
        for (const auto& element : *value.structure()) {
          TFF_TRY(MaterializeValue_(
              element, struct_pb->add_element()->mutable_value(), tasks));
        }
        return absl::OkStatus();
      }
    }
  }
};

}  // namespace

std::shared_ptr<Executor> CreateComposingExecutor(
    std::shared_ptr<Executor> server, std::vector<ComposingChild> children) {
  uint32_t total_clients = 0;
  for (const auto& child : children) {
    total_clients += child.num_clients();
  }
  return std::make_shared<ComposingExecutor>(
      std::move(server), std::move(children), total_clients);
}

}  // namespace tensorflow_federated

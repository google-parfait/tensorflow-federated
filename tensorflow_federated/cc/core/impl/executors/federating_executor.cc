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

#include "tensorflow_federated/cc/core/impl/executors/federating_executor.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
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

// NOTE: all variants must remain small and cheaply copyable (possibly via
// shared_ptr wrappers) so that `ExecutorValue` can be cheaply copied.
using UnplacedOrServer = std::shared_ptr<OwnedValueId>;
using Clients = std::shared_ptr<std::vector<std::shared_ptr<OwnedValueId>>>;
using Structure = std::shared_ptr<std::vector<ExecutorValue>>;
using ValueVariant =
    std::variant<UnplacedOrServer, Clients, Structure, enum FederatedIntrinsic>;

inline std::shared_ptr<OwnedValueId> ShareValueId(OwnedValueId&& id) {
  return std::make_shared<OwnedValueId>(std::move(id));
}

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

  inline static ExecutorValue Unplaced(UnplacedOrServer id) {
    return ExecutorValue(std::move(id), ValueType::UNPLACED);
  }
  inline const UnplacedOrServer& unplaced() const {
    return std::get<UnplacedOrServer>(value_);
  }
  inline static ExecutorValue Server(UnplacedOrServer id) {
    return ExecutorValue(std::move(id), ValueType::SERVER);
  }
  inline const UnplacedOrServer& server() const {
    return std::get<UnplacedOrServer>(value_);
  }
  inline const Clients& clients() const {
    return std::get<::tensorflow_federated::Clients>(value_);
  }
  inline static ExecutorValue Clients(Clients client_values) {
    return ExecutorValue(std::move(client_values), ValueType::CLIENTS);
  }
  // Convenience constructor from an un-shared_ptr vector.
  inline static ExecutorValue Clients(
      std::vector<std::shared_ptr<OwnedValueId>>&& client_values) {
    return Clients(std::make_shared<std::vector<std::shared_ptr<OwnedValueId>>>(
        std::move(client_values)));
  }
  inline const Structure& structure() const {
    return std::get<::tensorflow_federated::Structure>(value_);
  }
  inline static ExecutorValue Structure(Structure elements) {
    return ExecutorValue(std::move(elements), ValueType::STRUCTURE);
  }
  inline static ExecutorValue FederatedIntrinsic(FederatedIntrinsic intrinsic) {
    return ExecutorValue(intrinsic, ValueType::INTRINSIC);
  }
  inline enum FederatedIntrinsic intrinsic() const {
    return std::get<enum FederatedIntrinsic>(value_);
  }

  inline ValueType type() const { return type_; }

  ExecutorValue(ValueVariant value, ValueType type)
      : value_(std::move(value)), type_(type) {}

 private:
  ExecutorValue() = delete;
  ValueVariant value_;
  ValueType type_;
};

absl::Status CheckLenForUseAsArgument(const ExecutorValue& value,
                                      absl::string_view function_name,
                                      size_t len) {
  if (value.type() != ExecutorValue::ValueType::STRUCTURE) {
    return absl::InvalidArgumentError(
        absl::StrCat(function_name,
                     " expected a structural argument, found argument of type ",
                     value.type()));
  }
  if (value.structure()->size() != len) {
    return absl::InvalidArgumentError(absl::StrCat(function_name, " expected ",
                                                   len, " arguments, found ",
                                                   value.structure()->size()));
  }
  return absl::OkStatus();
}

class FederatingExecutor : public ExecutorBase<ExecutorValue> {
 public:
  explicit FederatingExecutor(std::shared_ptr<Executor> child,
                              uint32_t num_clients)
      : child_(child), num_clients_(num_clients) {}
  ~FederatingExecutor() override {
    // We must make sure to delete all of our OwnedValueIds, releasing them from
    // the child executor as well, before deleting the child executor.
    ClearTracked();
  }

 private:
  std::shared_ptr<Executor> child_;
  uint32_t num_clients_;

  const char* ExecutorName() final { return "FederatingExecutor"; }

  ExecutorValue ClientsAllEqualValue(
      const std::shared_ptr<OwnedValueId>& value) const {
    // All-equal-ness is not stored by this executor. Instead, we create
    // `num_clients_` non-all-equal references to the same value. This
    // prevents optimization of the uncommon "materialize a broadcasted
    // value" case, but allows for simpler handling of values throughout.
    return ExecutorValue::Clients(
        std::make_shared<std::vector<std::shared_ptr<OwnedValueId>>>(
            num_clients_, value));
  }

  Clients NewClients() {
    return ::tensorflow_federated::NewClients(num_clients_);
  }

  absl::StatusOr<ExecutorValue> CreateFederatedValue_(
      FederatedKind kind, const v0::Value_Federated& federated) {
    switch (kind) {
      case FederatedKind::SERVER: {
        return ExecutorValue::Server(
            ShareValueId(TFF_TRY(child_->CreateValue(federated.value(0)))));
      }
      case FederatedKind::CLIENTS: {
        Clients values = NewClients();
        for (const auto& value_pb : federated.value()) {
          values->emplace_back(
              ShareValueId(TFF_TRY(child_->CreateValue(value_pb)

                                       )));
        }
        return ExecutorValue::Clients(std::move(values));
      }
      case FederatedKind::CLIENTS_ALL_EQUAL: {
        return ClientsAllEqualValue(
            ShareValueId(TFF_TRY(child_->CreateValue(federated.value(0)))));
      }
    }
  }

  absl::StatusOr<std::shared_ptr<OwnedValueId>> Embed(
      const ExecutorValue& value) {
    switch (value.type()) {
      case ExecutorValue::ValueType::CLIENTS:
      case ExecutorValue::ValueType::SERVER: {
        return absl::InvalidArgumentError("Cannot embed a federated value.");
      }
      case ExecutorValue::ValueType::UNPLACED: {
        return value.unplaced();
      }
      case ExecutorValue::ValueType::INTRINSIC: {
        return absl::InvalidArgumentError(
            "Cannot embed a federated intrinsic.");
      }
      case ExecutorValue::ValueType::STRUCTURE: {
        std::vector<std::shared_ptr<OwnedValueId>> field_ids;
        std::vector<ValueId> unowned_field_ids;
        size_t num_fields = value.structure()->size();
        field_ids.reserve(num_fields);
        unowned_field_ids.reserve(num_fields);
        for (const auto& field_value : *value.structure()) {
          auto child_field_value_id = TFF_TRY(Embed(field_value));
          unowned_field_ids.push_back(child_field_value_id->ref());
          field_ids.push_back(std::move(child_field_value_id));
        }
        return ShareValueId(TFF_TRY(child_->CreateStruct(unowned_field_ids)));
      }
    }
  }

  absl::StatusOr<ExecutorValue> CreateExecutorValue(
      const v0::Value& value_pb) final {
    switch (value_pb.value_case()) {
      case v0::Value::kFederated: {
        const v0::Value_Federated& federated = value_pb.federated();
        auto kind = TFF_TRY(ValidateFederated(num_clients_, federated));
        return CreateFederatedValue_(kind, federated);
      }
      case v0::Value::kStruct: {
        auto elements = NewStructure();
        elements->reserve(value_pb.struct_().element_size());
        for (const auto& element_pb : value_pb.struct_().element()) {
          elements->emplace_back(
              TFF_TRY(CreateExecutorValue(element_pb.value())));
        }
        return ExecutorValue::Structure(std::move(elements));
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
        return ExecutorValue::Unplaced(
            ShareValueId(TFF_TRY(child_->CreateValue(value_pb))));
      }
    }
  }

  absl::StatusOr<ExecutorValue> CreateCall(
      ExecutorValue function, std::optional<ExecutorValue> argument) final {
    switch (function.type()) {
      case ExecutorValue::ValueType::CLIENTS:
      case ExecutorValue::ValueType::SERVER: {
        return absl::InvalidArgumentError("Cannot call a federated value.");
      }
      case ExecutorValue::ValueType::STRUCTURE: {
        return absl::InvalidArgumentError("Cannot call a structure.");
      }
      case ExecutorValue::ValueType::UNPLACED: {
        ValueId fn_id = function.unplaced()->ref();
        std::optional<std::shared_ptr<OwnedValueId>> arg_owner;
        std::optional<ValueId> arg_id = std::nullopt;
        if (argument.has_value()) {
          arg_owner = TFF_TRY(Embed(argument.value()));
          arg_id = arg_owner.value()->ref();
        }
        return ExecutorValue::Unplaced(
            ShareValueId(TFF_TRY(child_->CreateCall(fn_id, arg_id))));
      }
      case ExecutorValue::ValueType::INTRINSIC: {
        if (!argument.has_value()) {
          return absl::InvalidArgumentError(
              "no argument provided for federated intrinsic");
        }
        return CallFederatedIntrinsic(function.intrinsic(),
                                      std::move(argument.value()));
      }
    }
  }

  absl::StatusOr<ExecutorValue> CallFederatedIntrinsic(
      FederatedIntrinsic function, ExecutorValue arg) {
    switch (function) {
      case FederatedIntrinsic::VALUE_AT_CLIENTS: {
        return ClientsAllEqualValue(TFF_TRY(Embed(arg)));
      }
      case FederatedIntrinsic::VALUE_AT_SERVER: {
        return ExecutorValue::Server(TFF_TRY(Embed(arg)));
      }
      case FederatedIntrinsic::EVAL_AT_SERVER: {
        auto embedded = TFF_TRY(Embed(arg));
        return ExecutorValue::Server(ShareValueId(
            TFF_TRY(child_->CreateCall(embedded->ref(), std::nullopt))));
      }
      case FederatedIntrinsic::EVAL_AT_CLIENTS: {
        auto embedded = TFF_TRY(Embed(arg));
        Clients client_values = NewClients();
        for (int i = 0; i < num_clients_; i++) {
          client_values->emplace_back(ShareValueId(
              TFF_TRY(child_->CreateCall(embedded->ref(), std::nullopt))));
        }
        return ExecutorValue::Clients(std::move(client_values));
      }
      case FederatedIntrinsic::AGGREGATE: {
        TFF_TRY(CheckLenForUseAsArgument(arg, "federated_aggregate", 5));
        const auto& value = arg.structure()->at(0);
        const auto& zero = arg.structure()->at(1);
        auto zero_child_id = TFF_TRY(Embed(zero));
        const auto& accumulate = arg.structure()->at(2);
        auto accumulate_child_id = TFF_TRY(Embed(accumulate));
        // `merge` is unused (argument four).
        const auto& report = arg.structure()->at(4);
        auto report_child_id = TFF_TRY(Embed(report));
        if (value.type() != ExecutorValue::ValueType::CLIENTS) {
          return absl::InvalidArgumentError(
              "Cannot aggregate a value not placed at clients.");
        }
        std::optional<OwnedValueId> current_owner = std::nullopt;
        ValueId current = zero_child_id->ref();
        for (const auto& client_val_id : *value.clients()) {
          auto acc_arg =
              TFF_TRY(child_->CreateStruct({current, client_val_id->ref()}));
          current_owner =
              TFF_TRY(child_->CreateCall(accumulate_child_id->ref(), acc_arg));
          current = current_owner.value().ref();
        }
        auto result =
            TFF_TRY(child_->CreateCall(report_child_id->ref(), current));
        return ExecutorValue::Server(ShareValueId(std::move(result)));
      }
      case FederatedIntrinsic::BROADCAST: {
        if (arg.type() != ExecutorValue::ValueType::SERVER) {
          return absl::InvalidArgumentError(
              "Attempted to broadcast a value not placed at server.");
        }
        return ClientsAllEqualValue(arg.server());
      }
      case FederatedIntrinsic::MAP: {
        TFF_TRY(CheckLenForUseAsArgument(arg, "federated_map", 2));
        const auto& fn = arg.structure()->at(0);
        auto child_fn = TFF_TRY(Embed(fn));
        ValueId child_fn_ref = child_fn->ref();
        const auto& data = arg.structure()->at(1);
        if (data.type() == ExecutorValue::ValueType::CLIENTS) {
          Clients results = NewClients();
          for (int i = 0; i < num_clients_; i++) {
            auto client_arg = data.clients()->at(i)->ref();
            auto result = TFF_TRY(child_->CreateCall(child_fn_ref, client_arg));
            results->emplace_back(ShareValueId(std::move(result)));
          }
          return ExecutorValue::Clients(std::move(results));
        } else if (data.type() == ExecutorValue::ValueType::SERVER) {
          auto res =
              TFF_TRY(child_->CreateCall(child_fn_ref, data.server()->ref()));
          return ExecutorValue::Server(ShareValueId(std::move(res)));
        } else {
          return absl::InvalidArgumentError(
              "Attempted to map non-federated value.");
        }
      }
      case FederatedIntrinsic::ZIP: {
        TFF_TRY(CheckLenForUseAsArgument(arg, "federated_zip", 2));
        const auto& first = arg.structure()->at(0);
        const auto& second = arg.structure()->at(1);
        if (first.type() != second.type()) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Attempted to zip values with different placements: ",
              first.type(), " and ", second.type()));
        }
        if (first.type() == ExecutorValue::ValueType::CLIENTS) {
          Clients pairs = NewClients();
          for (int i = 0; i < num_clients_; i++) {
            ValueId first_id = first.clients()->at(i)->ref();
            ValueId second_id = second.clients()->at(i)->ref();
            auto pair = TFF_TRY(child_->CreateStruct({first_id, second_id}));
            pairs->emplace_back(ShareValueId(std::move(pair)));
          }
          return ExecutorValue::Clients(std::move(pairs));
        } else if (first.type() == ExecutorValue::ValueType::SERVER) {
          ValueId first_id = first.server()->ref();
          ValueId second_id = second.server()->ref();
          auto pair = TFF_TRY(child_->CreateStruct({first_id, second_id}));
          return ExecutorValue::Server(ShareValueId(std::move(pair)));
        } else {
          return absl::InvalidArgumentError(
              "Attempted to zip non-federated value.");
        }
      }
    }
  }

  absl::StatusOr<ExecutorValue> CreateStruct(
      std::vector<ExecutorValue> members) final {
    return ExecutorValue::Structure(
        std::make_shared<std::vector<ExecutorValue>>(std::move(members)));
  }

  absl::StatusOr<ExecutorValue> CreateSelection(ExecutorValue value,
                                                const uint32_t index) final {
    switch (value.type()) {
      case ExecutorValue::ValueType::CLIENTS:
      case ExecutorValue::ValueType::SERVER: {
        return absl::InvalidArgumentError("Cannot select from federated value");
      }
      case ExecutorValue::ValueType::UNPLACED: {
        ValueId ref = value.unplaced()->ref();
        return ExecutorValue::Unplaced(
            ShareValueId(TFF_TRY(child_->CreateSelection(ref, index))));
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
  }

  void CreateChildMaterializeTask(ValueId id, v0::Value* value_pb,
                                  ParallelTasks& tasks) {
    tasks.add_task([child = child_, id, value_pb]() {
      return child->Materialize(id, value_pb);
    });
  }

  absl::Status CreateMaterializeTasks(const ExecutorValue& value,
                                      v0::Value* value_pb,
                                      ParallelTasks& tasks) {
    switch (value.type()) {
      case ExecutorValue::ValueType::CLIENTS: {
        v0::Value_Federated* federated_pb = value_pb->mutable_federated();
        v0::FederatedType* type_pb = federated_pb->mutable_type();
        // All-equal-ness is not stored, so must be assumed to be false.
        // If the Python type system expects the value to be all-equal, it can
        // simply extract the first element in the list.
        type_pb->set_all_equal(false);
        *type_pb->mutable_placement()->mutable_value()->mutable_uri() =
            kClientsUri;
        for (const auto& client_value : *value.clients()) {
          CreateChildMaterializeTask(client_value->ref(),
                                     federated_pb->add_value(), tasks);
        }
        return absl::OkStatus();
      }
      case ExecutorValue::ValueType::UNPLACED: {
        CreateChildMaterializeTask(value.unplaced()->ref(), value_pb, tasks);
        return absl::OkStatus();
      }
      case ExecutorValue::ValueType::INTRINSIC: {
        return absl::UnimplementedError(
            "Materialization of federated intrinsics is not supported.");
      }
      case ExecutorValue::ValueType::SERVER: {
        v0::Value_Federated* federated_pb = value_pb->mutable_federated();
        v0::FederatedType* type_pb = federated_pb->mutable_type();
        // Server placement is assumed to be of cardinality one, and so must be
        // all-equal.
        type_pb->set_all_equal(true);
        *type_pb->mutable_placement()->mutable_value()->mutable_uri() =
            kServerUri;
        CreateChildMaterializeTask(value.server()->ref(),
                                   federated_pb->add_value(), tasks);
        return absl::OkStatus();
      }
      case ExecutorValue::ValueType::STRUCTURE: {
        v0::Value_Struct* struct_pb = value_pb->mutable_struct_();
        for (const auto& element : *value.structure()) {
          TFF_TRY(

              CreateMaterializeTasks(
                  element, struct_pb->add_element()->mutable_value(), tasks));
        }
        return absl::OkStatus();
      }
    }
  }

  absl::Status Materialize(ExecutorValue value, v0::Value* value_pb) {
    ParallelTasks tasks;
    TFF_TRY(CreateMaterializeTasks(value, value_pb, tasks));
    TFF_TRY(tasks.WaitAll());
    return absl::OkStatus();
  }
};

}  // namespace

absl::StatusOr<std::shared_ptr<Executor>> CreateFederatingExecutor(
    std::shared_ptr<Executor> child, const CardinalityMap& cardinalities) {
  int num_clients = TFF_TRY(NumClientsFromCardinalities(cardinalities));
  return std::make_shared<FederatingExecutor>(std::move(child), num_clients);
}

}  // namespace tensorflow_federated

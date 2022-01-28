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
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow_federated/cc/core/impl/executors/cardinalities.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/federated_intrinsics.h"
#include "tensorflow_federated/cc/core/impl/executors/status_macros.h"
#include "tensorflow_federated/cc/core/impl/executors/tensor_serialization.h"
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
using ValueVariant = absl::variant<UnplacedOrServer, Clients, Structure,
                                   enum FederatedIntrinsic>;

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

  inline static ExecutorValue CreateUnplaced(UnplacedOrServer id) {
    return ExecutorValue(std::move(id), ValueType::UNPLACED);
  }
  inline const UnplacedOrServer& unplaced() const {
    return absl::get<UnplacedOrServer>(value_);
  }
  inline static ExecutorValue CreateServerPlaced(UnplacedOrServer id) {
    return ExecutorValue(std::move(id), ValueType::SERVER);
  }
  inline const UnplacedOrServer& server() const {
    return absl::get<UnplacedOrServer>(value_);
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
  inline static ExecutorValue CreateFederatedIntrinsic(
      FederatedIntrinsic intrinsic) {
    return ExecutorValue(intrinsic, ValueType::INTRINSIC);
  }
  inline enum FederatedIntrinsic intrinsic() const {
    return absl::get<enum FederatedIntrinsic>(value_);
  }

  inline ValueType type() const { return type_; }

  absl::Status CheckArgumentType(ValueType expected_type,
                                 absl::string_view argument_identifier) const {
    if (type() == expected_type) {
      return absl::OkStatus();
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Expected ", argument_identifier,
                       " argument to have type ", expected_type,
                       ", but an argument of type ", type(), " was provided."));
    }
  }

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
  TFF_TRY(value.CheckArgumentType(ExecutorValue::ValueType::STRUCTURE,
                                  function_name));
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

  absl::string_view ExecutorName() final {
    static constexpr absl::string_view kExecutorName = "FederatingExecutor";
    return kExecutorName;
  }

  ExecutorValue ClientsAllEqualValue(
      const std::shared_ptr<OwnedValueId>& value) const {
    // All-equal-ness is not stored by this executor. Instead, we create
    // `num_clients_` non-all-equal references to the same value. This
    // prevents optimization of the uncommon "materialize a broadcasted
    // value" case, but allows for simpler handling of values throughout.
    return ExecutorValue::CreateClientsPlaced(
        std::make_shared<std::vector<std::shared_ptr<OwnedValueId>>>(
            num_clients_, value));
  }

  Clients NewClients() {
    return ::tensorflow_federated::NewClients(num_clients_);
  }

  absl::StatusOr<ExecutorValue> CreateFederatedValue(
      FederatedKind kind, const v0::Value_Federated& federated) {
    switch (kind) {
      case FederatedKind::SERVER: {
        return ExecutorValue::CreateServerPlaced(
            ShareValueId(TFF_TRY(child_->CreateValue(federated.value(0)))));
      }
      case FederatedKind::CLIENTS: {
        Clients values = NewClients();
        for (const auto& value_pb : federated.value()) {
          values->emplace_back(
              ShareValueId(TFF_TRY(child_->CreateValue(value_pb)

                                       )));
        }
        return ExecutorValue::CreateClientsPlaced(std::move(values));
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
        return CreateFederatedValue(kind, federated);
      }
      case v0::Value::kStruct: {
        auto elements = NewStructure();
        elements->reserve(value_pb.struct_().element_size());
        for (const auto& element_pb : value_pb.struct_().element()) {
          elements->emplace_back(
              TFF_TRY(CreateExecutorValue(element_pb.value())));
        }
        return ExecutorValue::CreateStructure(std::move(elements));
      }
      case v0::Value::kComputation: {
        if (value_pb.computation().has_intrinsic()) {
          return ExecutorValue::CreateFederatedIntrinsic(
              TFF_TRY(FederatedIntrinsicFromUri(
                  value_pb.computation().intrinsic().uri())));
        }
      }
        TF_FALLTHROUGH_INTENDED;
      default: {
        return ExecutorValue::CreateUnplaced(
            ShareValueId(TFF_TRY(child_->CreateValue(value_pb))));
      }
    }
  }

  absl::StatusOr<ExecutorValue> CreateCall(
      ExecutorValue function, absl::optional<ExecutorValue> argument) final {
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
        absl::optional<std::shared_ptr<OwnedValueId>> arg_owner;
        absl::optional<ValueId> arg_id = absl::nullopt;
        if (argument.has_value()) {
          arg_owner = TFF_TRY(Embed(argument.value()));
          arg_id = arg_owner.value()->ref();
        }
        return ExecutorValue::CreateUnplaced(
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

  // Embeds `arg` containing structures of server-placed values into the
  // `child_` executor.
  absl::StatusOr<std::shared_ptr<OwnedValueId>> ZipStructIntoServer(
      const ExecutorValue& arg) {
    switch (arg.type()) {
      case ExecutorValue::ValueType::SERVER: {
        return arg.server();
      }
      case ExecutorValue::ValueType::STRUCTURE: {
        std::vector<std::shared_ptr<OwnedValueId>> owned_element_ids;
        owned_element_ids.reserve(arg.structure()->size());
        for (const auto& element : *arg.structure()) {
          owned_element_ids.push_back(TFF_TRY(ZipStructIntoServer(element)));
        }
        std::vector<ValueId> element_ids;
        element_ids.reserve(arg.structure()->size());
        for (const auto& owned_id : owned_element_ids) {
          element_ids.push_back(owned_id->ref());
        }
        return ShareValueId(TFF_TRY(child_->CreateStruct(element_ids)));
      }
      default: {
        return absl::InvalidArgumentError(absl::StrCat(
            "Cannot `", kFederatedZipAtServerUri,
            "` a structure containing a value of kind ", arg.type()));
      }
    }
  }

  // Embeds `arg` containing structures of client-placed values into the
  // `child_` executor. The resulting structure on `child_` will contain all
  // values for the client corresponding to `client_index`.
  absl::StatusOr<std::shared_ptr<OwnedValueId>> ZipStructIntoClient(
      const ExecutorValue& arg, uint32_t client_index) {
    switch (arg.type()) {
      case ExecutorValue::ValueType::CLIENTS: {
        return (*arg.clients())[client_index];
      }
      case ExecutorValue::ValueType::STRUCTURE: {
        std::vector<std::shared_ptr<OwnedValueId>> owned_element_ids;
        owned_element_ids.reserve(arg.structure()->size());
        for (const auto& element : *arg.structure()) {
          owned_element_ids.push_back(
              TFF_TRY(ZipStructIntoClient(element, client_index)));
        }
        std::vector<ValueId> element_ids;
        element_ids.reserve(arg.structure()->size());
        for (const auto& owned_id : owned_element_ids) {
          element_ids.push_back(owned_id->ref());
        }
        return ShareValueId(TFF_TRY(child_->CreateStruct(element_ids)));
      }
      default: {
        return absl::InvalidArgumentError(absl::StrCat(
            "Cannot `", kFederatedZipAtClientsUri,
            "` a structure containing a value of kind ", arg.type()));
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
        return ExecutorValue::CreateServerPlaced(TFF_TRY(Embed(arg)));
      }
      case FederatedIntrinsic::EVAL_AT_SERVER: {
        auto embedded = TFF_TRY(Embed(arg));
        return ExecutorValue::CreateServerPlaced(ShareValueId(
            TFF_TRY(child_->CreateCall(embedded->ref(), absl::nullopt))));
      }
      case FederatedIntrinsic::EVAL_AT_CLIENTS: {
        auto embedded = TFF_TRY(Embed(arg));
        Clients client_values = NewClients();
        for (int i = 0; i < num_clients_; i++) {
          client_values->emplace_back(ShareValueId(
              TFF_TRY(child_->CreateCall(embedded->ref(), absl::nullopt))));
        }
        return ExecutorValue::CreateClientsPlaced(std::move(client_values));
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
        TFF_TRY(value.CheckArgumentType(ExecutorValue::ValueType::CLIENTS,
                                        "`federated_aggregate`'s `value`"));
        absl::optional<OwnedValueId> current_owner = absl::nullopt;
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
        return ExecutorValue::CreateServerPlaced(
            ShareValueId(std::move(result)));
      }
      case FederatedIntrinsic::BROADCAST: {
        TFF_TRY(arg.CheckArgumentType(ExecutorValue::ValueType::SERVER,
                                      "`federated_broadcast`"));
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
          return ExecutorValue::CreateClientsPlaced(std::move(results));
        } else if (data.type() == ExecutorValue::ValueType::SERVER) {
          auto res =
              TFF_TRY(child_->CreateCall(child_fn_ref, data.server()->ref()));
          return ExecutorValue::CreateServerPlaced(
              ShareValueId(std::move(res)));
        } else {
          return absl::InvalidArgumentError(
              "Attempted to map non-federated value.");
        }
      }
      case FederatedIntrinsic::SELECT: {
        TFF_TRY(CheckLenForUseAsArgument(arg, "federated_select", 4));
        const auto& keys = arg.structure()->at(0);
        // Argument two (`max_key`) is unused in this impl.
        const auto& server_val = arg.structure()->at(2);
        const auto& select_fn = arg.structure()->at(3);
        TFF_TRY(keys.CheckArgumentType(ExecutorValue::ValueType::CLIENTS,
                                       "`federated_select`'s `keys`"));
        const Clients& keys_child_ids = keys.clients();
        TFF_TRY(
            server_val.CheckArgumentType(ExecutorValue::ValueType::SERVER,
                                         "`federated_select`'s `server_val`"));
        ValueId server_val_child_id = server_val.server()->ref();
        TFF_TRY(
            select_fn.CheckArgumentType(ExecutorValue::ValueType::UNPLACED,
                                        "`federated_select`'s `select_fn`"));
        ValueId select_fn_child_id = select_fn.unplaced()->ref();
        return CallFederatedSelect(keys_child_ids, server_val_child_id,
                                   select_fn_child_id);
      }
      case FederatedIntrinsic::ZIP_AT_CLIENTS: {
        Clients results = NewClients();
        for (uint32_t i = 0; i < num_clients_; i++) {
          results->push_back(TFF_TRY(ZipStructIntoClient(arg, i)));
        }
        return ExecutorValue::CreateClientsPlaced(std::move(results));
      }
      case FederatedIntrinsic::ZIP_AT_SERVER: {
        return ExecutorValue::CreateServerPlaced(
            TFF_TRY(ZipStructIntoServer(arg)));
      }
    }
  }

  // A container for information about the keys in a `federated_select` round.
  struct KeyData {
    // The values of every key in a `federated_select`.
    // `all` contains the joined values of all `for_clients`.
    absl::flat_hash_set<int32_t> all;
    // The list of keys to slices requested for each client.
    // e.g. Client N's keys can be found at `for_clients[N]`.
    std::vector<std::vector<int32_t>> for_clients;
  };

  absl::StatusOr<ExecutorValue> CallFederatedSelect(
      const Clients& keys_child_ids, ValueId server_val_child_id,
      ValueId select_fn_child_id) {
    KeyData keys = TFF_TRY(MaterializeKeys(keys_child_ids));
    absl::flat_hash_map<int32_t, OwnedValueId> slice_for_key;
    slice_for_key.reserve(keys.all.size());
    for (int32_t key : keys.all) {
      slice_for_key.insert(
          {key, TFF_TRY(SelectSliceForKey(key, server_val_child_id,
                                          select_fn_child_id))});
    }
    v0::Value args_into_sequence_pb;
    args_into_sequence_pb.mutable_computation()->mutable_intrinsic()->set_uri(
        "args_into_sequence");
    OwnedValueId args_into_sequence_id =
        TFF_TRY(child_->CreateValue(args_into_sequence_pb));
    Clients client_datasets = NewClients();
    for (const auto& keys_for_client : keys.for_clients) {
      std::vector<ValueId> slice_ids_for_client;
      slice_ids_for_client.reserve(keys_for_client.size());
      for (int32_t key : keys_for_client) {
        slice_ids_for_client.push_back(slice_for_key.at(key).ref());
      }
      OwnedValueId slices = TFF_TRY(child_->CreateStruct(slice_ids_for_client));
      OwnedValueId dataset =
          TFF_TRY(child_->CreateCall(args_into_sequence_id, slices));
      client_datasets->push_back(ShareValueId(std::move(dataset)));
    }
    return ExecutorValue::CreateClientsPlaced(std::move(client_datasets));
  }

  absl::StatusOr<KeyData> MaterializeKeys(const Clients& keys_child_ids) {
    KeyData keys;
    keys.for_clients.reserve(keys_child_ids->size());
    for (const auto& keys_child_id : *keys_child_ids) {
      // TODO(b/209504748) Make federating_executor value a future so that these
      // materialize calls don't block.
      v0::Value keys_for_client_pb =
          TFF_TRY(child_->Materialize(keys_child_id->ref()));
      tensorflow::Tensor keys_for_client_tensor =
          TFF_TRY(DeserializeTensorValue(keys_for_client_pb));
      if (keys_for_client_tensor.dtype() != tensorflow::DT_INT32) {
        return absl::InvalidArgumentError(
            absl::StrCat("Expected int32_t key, found key of tensor dtype ",
                         keys_for_client_tensor.dtype()));
      }
      if (keys_for_client_tensor.dims() != 1) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Expected key tensor to be rank one, but found tensor of rank ",
            keys_for_client_tensor.dims()));
      }
      int64_t num_keys = keys_for_client_tensor.NumElements();
      std::vector<int32_t> keys_for_client;
      keys_for_client.reserve(num_keys);
      auto keys_for_client_eigen = keys_for_client_tensor.flat<int32_t>();
      for (int64_t i = 0; i < num_keys; i++) {
        int32_t key = keys_for_client_eigen(i);
        keys_for_client.push_back(key);
        keys.all.insert(key);
      }
      keys.for_clients.push_back(std::move(keys_for_client));
    }
    return keys;
  }

  absl::StatusOr<OwnedValueId> SelectSliceForKey(int32_t key,
                                                 ValueId server_val_child_id,
                                                 ValueId select_fn_child_id) {
    v0::Value key_pb;
    TFF_TRY(SerializeTensorValue(tensorflow::Tensor(key), &key_pb));
    OwnedValueId key_id = TFF_TRY(child_->CreateValue(key_pb));
    OwnedValueId arg_id =
        TFF_TRY(child_->CreateStruct({server_val_child_id, key_id}));
    return TFF_TRY(child_->CreateCall(select_fn_child_id, arg_id));
  }

  absl::StatusOr<ExecutorValue> CreateStruct(
      std::vector<ExecutorValue> members) final {
    return ExecutorValue::CreateStructure(
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
        return ExecutorValue::CreateUnplaced(
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
        type_pb->mutable_placement()->mutable_value()->mutable_uri()->assign(
            kClientsUri.data(), kClientsUri.size());
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
        type_pb->mutable_placement()->mutable_value()->mutable_uri()->assign(
            kServerUri.data(), kServerUri.size());
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

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
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
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
  explicit UnplacedInner(std::shared_ptr<v0::Value> proto)
      : proto_(std::move(proto)) {}
  explicit UnplacedInner(v0::Value proto)
      : proto_(std::make_shared<v0::Value>(std::move(proto))) {}

  explicit UnplacedInner(std::shared_ptr<OwnedValueId> embedded)
      : embedded_(std::move(embedded)) {}
  explicit UnplacedInner(OwnedValueId embedded)
      : embedded_(std::make_shared<OwnedValueId>(std::move(embedded))) {}

  // NOTE: all constructors must set `proto` or optionally `embedded` value.
  // Internals assume that proto value is set.

  // Returns the underlying `Proto` value without attempting to create one via
  // a `Materialize` call.
  std::optional<absl::StatusOr<std::shared_ptr<v0::Value>>> GetProto() {
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
        return ExtractProto();
      }
    }
    {
      // Try to grab the proto if it has been created since the last lock was
      // released.
      absl::WriterMutexLock lock(&mutex_);
      if (proto_.has_value()) {
        return ExtractProto();
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
      return ExtractProto();
    }
  }

  // Gets the embedded representation of the inner value.
  // May trigger a remote `CreateValue` call.
  absl::StatusOr<std::shared_ptr<OwnedValueId>> Embedded(Executor& server) {
    {
      // Try to grab the embedded ID if it already exists.
      absl::ReaderMutexLock lock(&mutex_);
      if (embedded_.has_value()) {
        return ExtractEmbedded();
      }
    }
    {
      // Try to grab the embedded ID if it has been created since the last lock
      // was released.
      absl::WriterMutexLock lock(&mutex_);
      if (embedded_.has_value()) {
        return ExtractEmbedded();
      }
      // Embed the proto into the underlying executor to get an ID.
      auto id_or_status = server.CreateValue(*proto_.value().value());
      if (id_or_status.ok()) {
        embedded_ = ShareValueId(std::move(id_or_status.value()));
      } else {
        embedded_ = std::move(id_or_status.status());
      }
      return ExtractEmbedded();
    }
  }

 private:
  absl::StatusOr<std::shared_ptr<v0::Value>> ExtractProto() const
      ABSL_SHARED_LOCKS_REQUIRED(mutex_) {
    if (proto_.value().ok()) {
      return proto_.value().value();
    } else {
      return proto_->status();
    }
  }

  absl::StatusOr<std::shared_ptr<OwnedValueId>> ExtractEmbedded() const
      ABSL_SHARED_LOCKS_REQUIRED(mutex_) {
    if (embedded_.value().ok()) {
      return embedded_.value().value();
    } else {
      return embedded_->status();
    }
  }

  absl::Mutex mutex_;
  std::optional<absl::StatusOr<std::shared_ptr<v0::Value>>> proto_
      ABSL_GUARDED_BY(mutex_);
  std::optional<absl::StatusOr<std::shared_ptr<OwnedValueId>>> embedded_
      ABSL_GUARDED_BY(mutex_);
};

// NOTE: all variants must remain small and cheaply copyable (possibly via
// shared_ptr wrappers) so that `ExecutorValue` can be cheaply copied.
using Unplaced = std::shared_ptr<UnplacedInner>;
using Server = std::shared_ptr<OwnedValueId>;
using Clients = std::shared_ptr<std::vector<std::shared_ptr<OwnedValueId>>>;
using Structure = std::shared_ptr<std::vector<ExecutorValue>>;
using ValueVariant =
    std::variant<Unplaced, Server, Clients, Structure, enum FederatedIntrinsic>;

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

  inline static ExecutorValue CreateUnplaced(
      ::tensorflow_federated::Unplaced id) {
    return ExecutorValue(std::move(id), ValueType::UNPLACED);
  }
  inline const Unplaced& unplaced() const { return std::get<Unplaced>(value_); }
  inline static ExecutorValue CreateServerPlaced(Server id) {
    return ExecutorValue(std::move(id), ValueType::SERVER);
  }
  inline const Server& server() const { return std::get<Server>(value_); }
  inline const Clients& clients() const {
    return std::get<::tensorflow_federated::Clients>(value_);
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
    return std::get<::tensorflow_federated::Structure>(value_);
  }
  inline static ExecutorValue CreateStructure(Structure elements) {
    return ExecutorValue(std::move(elements), ValueType::STRUCTURE);
  }
  inline static ExecutorValue CreateFederatedIntrinsic(
      FederatedIntrinsic intrinsic) {
    return ExecutorValue(intrinsic, ValueType::INTRINSIC);
  }
  inline enum FederatedIntrinsic intrinsic() const {
    return std::get<enum FederatedIntrinsic>(value_);
  }

  inline ValueType type() const { return type_; }

  absl::Status CheckArgumentType(ValueType expected_type,
                                 std::string_view argument_identifier) const {
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
                                      std::string_view function_name,
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
  explicit FederatingExecutor(std::shared_ptr<Executor> server_child,
                              std::shared_ptr<Executor> client_child,
                              uint32_t num_clients)
      : server_child_(server_child),
        client_child_(client_child),
        num_clients_(num_clients) {}
  ~FederatingExecutor() override {
    // We must make sure to delete all of our OwnedValueIds, releasing them from
    // the child executor as well, before deleting the child executor.
    ClearTracked();
  }

 private:
  std::shared_ptr<Executor> server_child_;
  std::shared_ptr<Executor> client_child_;
  uint32_t num_clients_;

  std::string_view ExecutorName() final {
    static constexpr std::string_view kExecutorName = "FederatingExecutor";
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
        return ExecutorValue::CreateServerPlaced(ShareValueId(
            TFF_TRY(server_child_->CreateValue(federated.value(0)))));
      }
      case FederatedKind::CLIENTS: {
        Clients values = NewClients();
        for (const auto& value_pb : federated.value()) {
          values->emplace_back(
              ShareValueId(TFF_TRY(client_child_->CreateValue(value_pb))));
        }
        return ExecutorValue::CreateClientsPlaced(std::move(values));
      }
      case FederatedKind::CLIENTS_ALL_EQUAL: {
        return ClientsAllEqualValue(ShareValueId(
            TFF_TRY(client_child_->CreateValue(federated.value(0)))));
      }
    }
  }
  bool is_server_child(Executor* child) const {
    return child == server_child_.get();
  }
  absl::StatusOr<std::shared_ptr<OwnedValueId>> Embed(
      const ExecutorValue& value, std::shared_ptr<Executor> child) {
    switch (value.type()) {
      case ExecutorValue::ValueType::CLIENTS:
      case ExecutorValue::ValueType::SERVER: {
        return absl::InvalidArgumentError("Cannot embed a federated value.");
      }
      case ExecutorValue::ValueType::UNPLACED: {
        // server_child_ also handles all unplaced value processing, it is not
        // strictly only the server placed value executor. Thus, extract proto
        // from the value using server_child.
        if (is_server_child(child.get())) {
          return TFF_TRY(value.unplaced()->Embedded(*child));
        } else {
          // In some cases Unplaced value could have only embedded value id on
          // the server. E.g. when result of CreateCall.
          // Use value.unplaced()->Proto to materialize value on server to
          // create a new value on the passed child executer.
          auto value_pb = TFF_TRY(value.unplaced()->Proto(*server_child_));
          return ShareValueId(TFF_TRY(child->CreateValue(*value_pb)));
        }
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
          auto child_field_value_id = TFF_TRY(Embed(field_value, child));
          unowned_field_ids.push_back(child_field_value_id->ref());
          field_ids.push_back(std::move(child_field_value_id));
        }
        return ShareValueId(TFF_TRY(child->CreateStruct(unowned_field_ids)));
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
          auto intrinsic = FederatedIntrinsicFromUri(
              value_pb.computation().intrinsic().uri());
          if (intrinsic.ok()) {
            return ExecutorValue::CreateFederatedIntrinsic(intrinsic.value());
          }
          // If the intrinsic is not federated (e.g. sequence_*) fall-through to
          // the default block below, passing the intrinsic to the child
          // executor.
        }
      }
        TF_FALLTHROUGH_INTENDED;
      default: {
        return ExecutorValue::CreateUnplaced(
            std::make_shared<UnplacedInner>(value_pb));
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
        ValueId fn_id =
            function.unplaced()->Embedded(*server_child_).value()->ref();
        std::optional<std::shared_ptr<OwnedValueId>> arg_owner;
        std::optional<ValueId> arg_id = std::nullopt;
        if (argument.has_value()) {
          arg_owner = TFF_TRY(Embed(argument.value(), server_child_));
          arg_id = arg_owner.value()->ref();
        }
        return ExecutorValue::CreateUnplaced(std::make_shared<UnplacedInner>(
            ShareValueId(TFF_TRY(server_child_->CreateCall(fn_id, arg_id)))));
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
        return ShareValueId(TFF_TRY(server_child_->CreateStruct(element_ids)));
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
        return ShareValueId(TFF_TRY(client_child_->CreateStruct(element_ids)));
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
        return ClientsAllEqualValue(TFF_TRY(Embed(arg, client_child_)));
      }
      case FederatedIntrinsic::VALUE_AT_SERVER: {
        return ExecutorValue::CreateServerPlaced(
            TFF_TRY(Embed(arg, server_child_)));
      }
      case FederatedIntrinsic::EVAL_AT_SERVER: {
        auto embedded = TFF_TRY(Embed(arg, server_child_));
        return ExecutorValue::CreateServerPlaced(ShareValueId(
            TFF_TRY(server_child_->CreateCall(embedded->ref(), std::nullopt))));
      }
      case FederatedIntrinsic::EVAL_AT_CLIENTS: {
        auto embedded = TFF_TRY(Embed(arg, client_child_));
        Clients client_values = NewClients();
        for (int i = 0; i < num_clients_; i++) {
          client_values->emplace_back(ShareValueId(TFF_TRY(
              client_child_->CreateCall(embedded->ref(), std::nullopt))));
        }
        return ExecutorValue::CreateClientsPlaced(std::move(client_values));
      }
      case FederatedIntrinsic::AGGREGATE: {
        auto traceme = Trace("CallFederatedAggregate");
        TFF_TRY(CheckLenForUseAsArgument(arg, "federated_aggregate", 5));
        const auto& value = arg.structure()->at(0);
        const auto& zero = arg.structure()->at(1);
        v0::Value zero_val;

        ParallelTasks tasks;
        TFF_TRY(CreateMaterializeTasks(zero, &zero_val, tasks));
        TFF_TRY(tasks.WaitAll());

        const auto& accumulate = arg.structure()->at(2);
        auto accumulate_val_or = accumulate.unplaced()->GetProto();
        if (!accumulate_val_or.has_value()) {
          return absl::InvalidArgumentError(
              "Failed to get accumulate function.");
        }
        // `merge` is unused (argument four).
        const auto& report = arg.structure()->at(4);
        auto report_child_id = TFF_TRY(Embed(report, server_child_));
        TFF_TRY(value.CheckArgumentType(ExecutorValue::ValueType::CLIENTS,
                                        "`federated_aggregate`'s `value`"));
        std::optional<OwnedValueId> current_owner = std::nullopt;
        auto zero_val_id_owner = TFF_TRY(client_child_->CreateValue(zero_val));
        ValueId current = zero_val_id_owner.ref();
        auto accumulate_child_id = TFF_TRY(
            client_child_->CreateValue(*(accumulate_val_or.value()->get())));
        for (const auto& client_val_id : *value.clients()) {
          auto acc_arg = TFF_TRY(
              client_child_->CreateStruct({current, client_val_id->ref()}));
          current_owner =
              TFF_TRY(client_child_->CreateCall(accumulate_child_id, acc_arg));
          current = current_owner.value().ref();
        }
        v0::Value result_val;
        TFF_TRY(client_child_->Materialize(current, &result_val));

        auto res = TFF_TRY(server_child_->CreateValue(result_val));
        auto result =
            TFF_TRY(server_child_->CreateCall(report_child_id->ref(), res));
        return ExecutorValue::CreateServerPlaced(
            ShareValueId(std::move(result)));
      }
      case FederatedIntrinsic::BROADCAST: {
        auto traceme = Trace("CallFederatedBroadcast");
        TFF_TRY(arg.CheckArgumentType(ExecutorValue::ValueType::SERVER,
                                      "`federated_broadcast`"));
        v0::Value server_val;
        TFF_TRY(server_child_->Materialize(arg.server()->ref(), &server_val));
        return ClientsAllEqualValue(
            ShareValueId(TFF_TRY(client_child_->CreateValue(server_val))));
      }
      case FederatedIntrinsic::MAP: {
        auto traceme = Trace("CallFederatedMap");
        TFF_TRY(CheckLenForUseAsArgument(arg, "federated_map", 2));
        const auto& fn = arg.structure()->at(0);
        const auto& data = arg.structure()->at(1);
        if (data.type() == ExecutorValue::ValueType::CLIENTS) {
          if (fn.type() != ExecutorValue::ValueType::UNPLACED) {
            return absl::InvalidArgumentError(absl::StrCat(
                "Function argument "
                "for MAP intrinsic must be unplaced. Found of type ",
                fn.type()));
          }
          auto child_fn_val = fn.unplaced()->GetProto();
          if (!child_fn_val.has_value()) {
            return absl::InternalError(
                "Unable to extract function proto to place on client. This "
                "function has probably already been embedded in a child "
                "executor for some reason.");
          }
          auto child_fn = TFF_TRY(
              client_child_->CreateValue(*(child_fn_val.value()->get())));
          Clients results = NewClients();
          for (int i = 0; i < num_clients_; i++) {
            auto client_arg = data.clients()->at(i)->ref();
            auto result =
                TFF_TRY(client_child_->CreateCall(child_fn, client_arg));
            results->emplace_back(ShareValueId(std::move(result)));
          }
          return ExecutorValue::CreateClientsPlaced(std::move(results));
        } else if (data.type() == ExecutorValue::ValueType::SERVER) {
          auto child_fn = TFF_TRY(Embed(fn, server_child_));
          auto res = TFF_TRY(
              server_child_->CreateCall(child_fn->ref(), data.server()->ref()));
          return ExecutorValue::CreateServerPlaced(
              ShareValueId(std::move(res)));
        } else {
          return absl::InvalidArgumentError(
              "Attempted to map non-federated value.");
        }
      }
      case FederatedIntrinsic::SELECT: {
        auto traceme = Trace("CallFederatedSelect");
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
        ValueId select_fn_child_id =
            select_fn.unplaced()->Embedded(*server_child_).value()->ref();
        return CallFederatedSelect(keys_child_ids, server_val_child_id,
                                   select_fn_child_id);
      }
      case FederatedIntrinsic::ZIP_AT_CLIENTS: {
        auto traceme = Trace("CallIntrinsicZipClients");
        Clients results = NewClients();
        for (uint32_t i = 0; i < num_clients_; i++) {
          results->push_back(TFF_TRY(ZipStructIntoClient(arg, i)));
        }
        return ExecutorValue::CreateClientsPlaced(std::move(results));
      }
      case FederatedIntrinsic::ZIP_AT_SERVER: {
        auto traceme = Trace("CallIntrinsicZipServer");
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
        TFF_TRY(server_child_->CreateValue(args_into_sequence_pb));
    Clients client_datasets = NewClients();
    for (const auto& keys_for_client : keys.for_clients) {
      std::vector<ValueId> slice_ids_for_client;
      slice_ids_for_client.reserve(keys_for_client.size());
      for (int32_t key : keys_for_client) {
        slice_ids_for_client.push_back(slice_for_key.at(key).ref());
      }
      OwnedValueId slices =
          TFF_TRY(server_child_->CreateStruct(slice_ids_for_client));
      OwnedValueId dataset =
          TFF_TRY(server_child_->CreateCall(args_into_sequence_id, slices));
      v0::Value dataset_pb;
      TFF_TRY(server_child_->Materialize(dataset.ref(), &dataset_pb));
      client_datasets->push_back(
          ShareValueId(TFF_TRY(client_child_->CreateValue(dataset_pb))));
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
          TFF_TRY(client_child_->Materialize(keys_child_id->ref()));
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
    OwnedValueId key_id = TFF_TRY(server_child_->CreateValue(key_pb));
    OwnedValueId arg_id =
        TFF_TRY(server_child_->CreateStruct({server_val_child_id, key_id}));
    return TFF_TRY(server_child_->CreateCall(select_fn_child_id, arg_id));
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
        auto id = TFF_TRY(value.unplaced()->Embedded(*server_child_));
        return ExecutorValue::CreateUnplaced(
            std::make_shared<UnplacedInner>(ShareValueId(
                TFF_TRY(server_child_->CreateSelection(id->ref(), index)))));
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

  absl::Status CreateChildMaterializeTask(ValueId id, v0::Value* value_pb,
                                          std::shared_ptr<Executor> child,
                                          ParallelTasks& tasks) {
    return tasks.add_task(
        [child, id, value_pb]() { return child->Materialize(id, value_pb); });
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
          TFF_TRY(CreateChildMaterializeTask(client_value->ref(),
                                             federated_pb->add_value(),
                                             client_child_, tasks));
        }
        return absl::OkStatus();
      }
      case ExecutorValue::ValueType::UNPLACED: {
        return tasks.add_task(
            [value = std::move(value), value_pb, server = server_child_]() {
              *value_pb = *TFF_TRY(value.unplaced()->Proto(*server));
              return absl::OkStatus();
            });
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
        return CreateChildMaterializeTask(value.server()->ref(),
                                          federated_pb->add_value(),
                                          server_child_, tasks);
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

  absl::Status Materialize(ExecutorValue value, v0::Value* value_pb) override {
    ParallelTasks tasks;
    TFF_TRY(CreateMaterializeTasks(value, value_pb, tasks));
    TFF_TRY(tasks.WaitAll());
    return absl::OkStatus();
  }
};

}  // namespace

absl::StatusOr<std::shared_ptr<Executor>> CreateFederatingExecutor(
    std::shared_ptr<Executor> server_child,
    std::shared_ptr<Executor> client_child,
    const CardinalityMap& cardinalities) {
  int num_clients = TFF_TRY(NumClientsFromCardinalities(cardinalities));
  return std::make_shared<FederatingExecutor>(
      std::move(server_child), std::move(client_child), num_clients);
}

}  // namespace tensorflow_federated

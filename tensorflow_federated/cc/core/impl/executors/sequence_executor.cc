/* Copyright 2022, The TensorFlow Federated Authors.

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

#include "tensorflow_federated/cc/core/impl/executors/sequence_executor.h"

#include <functional>
#include <future>  // NOLINT
#include <memory>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/status/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow_federated/cc/core/impl/executors/dataset_conversions.h"
#include "tensorflow_federated/cc/core/impl/executors/executor.h"
#include "tensorflow_federated/cc/core/impl/executors/sequence_intrinsics.h"
#include "tensorflow_federated/cc/core/impl/executors/tensor_serialization.h"
#include "tensorflow_federated/cc/core/impl/executors/threading.h"
#include "tensorflow_federated/proto/v0/executor.pb.h"

namespace tensorflow_federated {

namespace {

using Embedded = std::shared_ptr<OwnedValueId>;

Embedded ShareValueId(OwnedValueId id) {
  return std::make_shared<OwnedValueId>(std::move(id));
}

// Internal interface representing an iterable value embedded
// in the sequence executor.

// TODO(b/217763470): More implementation on the way for this class; it
// will need in particular to support iterating over its elements. For now we
// know this class will need to store a proto, for potential lazy embedding in a
// child executor.
class Sequence {
 public:
  explicit Sequence(v0::Value proto) : value_proto_(std::move(proto)) {}

  v0::Value proto() { return value_proto_; }

 private:
  v0::Value value_proto_;
};

class SequenceExecutorValue;

using ValueVariant =
    absl::variant<Embedded, SequenceIntrinsic, std::shared_ptr<Sequence>,
                  std::shared_ptr<std::vector<SequenceExecutorValue>>>;

class SequenceExecutorValue {
 public:
  enum class ValueType { EMBEDDED, INTRINSIC, SEQUENCE, STRUCT };

  inline static SequenceExecutorValue CreateEmbedded(Embedded id) {
    return SequenceExecutorValue(id);
  }
  inline static SequenceExecutorValue CreateIntrinsic(
      SequenceIntrinsic intrinsic) {
    return SequenceExecutorValue(std::move(intrinsic));
  }
  inline static SequenceExecutorValue CreateSequence(v0::Value sequence_value) {
    return SequenceExecutorValue(std::make_shared<Sequence>(sequence_value));
  }
  inline static SequenceExecutorValue CreateStruct(
      std::vector<SequenceExecutorValue>&& structure) {
    return SequenceExecutorValue(
        std::make_shared<std::vector<SequenceExecutorValue>>(structure));
  }

  inline ValueType type() const {
    if (absl::holds_alternative<Embedded>(value_)) {
      return ValueType::EMBEDDED;
    } else if (absl::holds_alternative<SequenceIntrinsic>(value_)) {
      return ValueType::INTRINSIC;
    } else if (absl::holds_alternative<std::shared_ptr<Sequence>>(value_)) {
      return ValueType::SEQUENCE;
    } else {
      return ValueType::STRUCT;
    }
  }

  inline Embedded embedded() { return absl::get<Embedded>(value_); }
  inline std::shared_ptr<Sequence> sequence_value() {
    return absl::get<std::shared_ptr<Sequence>>(value_);
  }
  inline std::shared_ptr<std::vector<SequenceExecutorValue>> struct_value() {
    return absl::get<std::shared_ptr<std::vector<SequenceExecutorValue>>>(
        value_);
  }

  explicit SequenceExecutorValue(ValueVariant value)
      : value_(std::move(value)) {}

 private:
  SequenceExecutorValue() = delete;
  ValueVariant value_;
};

// We return futures since pulling elements from a sequence may be slow, and
// otherwise would block.
using ValueFuture = std::shared_future<absl::StatusOr<SequenceExecutorValue>>;

class SequenceExecutor : public ExecutorBase<ValueFuture> {
 public:
  explicit SequenceExecutor(std::shared_ptr<Executor> target_executor)
      : target_executor_(target_executor) {}
  ~SequenceExecutor() override {}

  absl::string_view ExecutorName() final { return "SequenceExecutor"; }

  absl::StatusOr<ValueFuture> CreateExecutorValue(
      const v0::Value& value_pb) final {
    switch (value_pb.value_case()) {
      case v0::Value::kSequence: {
        // We store the sequence value as a proto in the SequenceExecutor for
        // lazy embedding in the lower-level executor if needed, e.g. in
        // response to a CreateCall, or construction of an iterable from this
        // sequence in the sequence executor itself.
        return ReadyFuture(SequenceExecutorValue::CreateSequence(value_pb));
      }
      case v0::Value::kComputation: {
        if (value_pb.computation().has_intrinsic()) {
          absl::string_view intrinsic_uri =
              value_pb.computation().intrinsic().uri();
          absl::StatusOr<SequenceIntrinsic> intrinsic_or_status =
              SequenceIntrinsicFromUri(intrinsic_uri);
          if (intrinsic_or_status.ok()) {
            return ReadyFuture(SequenceExecutorValue::CreateIntrinsic(
                SequenceIntrinsic(intrinsic_or_status.value())));
          }
        }
      }
        // We fall-through here to let intrinsics possibly meant for lower-level
        // executors pass through.
        ABSL_FALLTHROUGH_INTENDED;
      default:
        return ReadyFuture(SequenceExecutorValue::CreateEmbedded(
            ShareValueId(TFF_TRY(target_executor_->CreateValue(value_pb)))));
    }
  }

  absl::StatusOr<ValueFuture> CreateCall(
      ValueFuture function, absl::optional<ValueFuture> argument) final {
    return absl::UnimplementedError("Not yet implemented.");
  }

  absl::StatusOr<ValueFuture> CreateStruct(
      std::vector<ValueFuture> members) final {
    std::function<absl::StatusOr<SequenceExecutorValue>(
        std::vector<SequenceExecutorValue> &&)>
        mapping_fn = [executor = this->target_executor_](
                         std::vector<SequenceExecutorValue>&& member_elems) {
          return SequenceExecutorValue::CreateStruct(std::move(member_elems));
        };
    return Map(std::move(members), mapping_fn);
  }

  absl::StatusOr<ValueFuture> CreateSelection(ValueFuture value,
                                              const uint32_t index) final {
    std::function<absl::StatusOr<SequenceExecutorValue>(
        std::vector<SequenceExecutorValue> &&)>
        mapping_fn = [executor = this->target_executor_,
                      index](std::vector<SequenceExecutorValue>&& source_vector)
        -> absl::StatusOr<SequenceExecutorValue> {
      // We know there is exactly one element here since we will construct the
      // future-vector which supplies the argument below.
      SequenceExecutorValue source = source_vector.at(0);

      switch (source.type()) {
        case SequenceExecutorValue::ValueType::EMBEDDED: {
          return SequenceExecutorValue::CreateEmbedded(ShareValueId(TFF_TRY(
              executor->CreateSelection(source.embedded()->ref(), index))));
        }
        case SequenceExecutorValue::ValueType::STRUCT: {
          auto struct_val = source.struct_value();

          if (index < 0 || index >= struct_val->size()) {
            return absl::InvalidArgumentError(
                absl::StrCat("Attempted to select an element out-of-bounds of "
                             "the underlying "
                             "structure; structure is of length ",
                             struct_val->size(),
                             ", but attempted to select element ", index));
          }
          return struct_val->at(index);
        }
        default:
          return absl::InvalidArgumentError(
              "Can only select from embedded or structure values.");
      }
    };
    return Map(std::vector<ValueFuture>({value}), mapping_fn);
  }

  absl::Status Materialize(ValueFuture value, v0::Value* value_pb) final {
    SequenceExecutorValue exec_value = TFF_TRY(Wait(value));
    switch (exec_value.type()) {
      case SequenceExecutorValue::ValueType::EMBEDDED:
      case SequenceExecutorValue::ValueType::SEQUENCE:
      case SequenceExecutorValue::ValueType::STRUCT: {
        Embedded embedded_value = TFF_TRY(Embed(exec_value));
        TFF_TRY(target_executor_->Materialize(embedded_value->ref(), value_pb));
        return absl::OkStatus();
      }
      case SequenceExecutorValue::ValueType::INTRINSIC: {
        return absl::UnimplementedError(
            "Materialization of sequence intrinsics is not supported.");
      }
    }
    return absl::UnimplementedError("Not implemented yet!");
  }

 private:
  absl::StatusOr<std::shared_ptr<OwnedValueId>> Embed(
      SequenceExecutorValue val) {
    switch (val.type()) {
      case SequenceExecutorValue::ValueType::EMBEDDED:
        return val.embedded();
      case SequenceExecutorValue::ValueType::SEQUENCE:
        // Once we extend Sequence to handle iteration, we will need to check
        // that this is a proto-represented sequence, and not e.g. a sequence
        // after computing a sequence map with some embedded function.
        return ShareValueId(TFF_TRY(
            target_executor_->CreateValue(val.sequence_value()->proto())));
      case SequenceExecutorValue::ValueType::STRUCT: {
        auto struct_val = val.struct_value();
        std::vector<Embedded> owned_ids;
        std::vector<ValueId> refs;
        owned_ids.reserve(struct_val->size());
        refs.reserve(struct_val->size());
        for (const SequenceExecutorValue& elem : *struct_val) {
          Embedded embedded_val = TFF_TRY(Embed(elem));
          owned_ids.emplace_back(embedded_val);
          refs.emplace_back(embedded_val->ref());
        }
        return ShareValueId(TFF_TRY(target_executor_->CreateStruct(refs)));
      }
      default:
        return absl::UnimplementedError(
            "Embed only implemented for Embedded, Sequence, and Struct "
            "values.");
    }
  }
  std::shared_ptr<Executor> target_executor_;
};

}  // namespace

std::shared_ptr<Executor> CreateSequenceExecutor(
    std::shared_ptr<Executor> target_executor) {
  return std::make_unique<SequenceExecutor>(target_executor);
}

}  // namespace tensorflow_federated
